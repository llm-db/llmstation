import time
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import peft.tuners.lora.layer as _lora_layer

torch.manual_seed(42)
torch.cuda.manual_seed(42)

model_name = "meta-llama/Llama-3.2-3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_name, dtype=torch.bfloat16, device_map="auto",
    attn_implementation="flash_attention_2",
)

# 2 LoRA adapters: ft (train) + infer (shared by both inference requests)
lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.0)
model = get_peft_model(model, lora_config, adapter_name="ft")
model.add_adapter("infer", lora_config)
model.set_adapter("ft")  # only ft adapter requires grad
model.print_trainable_parameters()

# ─────────────────────────────────────────────────────────────────────────
# Gather-BMM: batch N inference requests through ONE base-model forward pass.
# The per-request adapter assignment is decided per step (real serving has
# varying request counts and adapter mixes), so A/B are stacked ON THE FLY
# inside the patched LoRA forward using the current step's adapter list.
# Input is (N,L,H) and:
#   base_out = x @ W_base^T                (single cuBLAS, shared base)
#   lora_out = bmm(bmm(x, A^T), B^T) * α    (batched per-sample BMM)
# Set GBMM_STEP_ADAPTERS = [name, name, ...] before each decode; reset to
# None for training (so PEFT's normal forward runs with the ft adapter).
# ─────────────────────────────────────────────────────────────────────────
GBMM_STEP_ADAPTERS = None  # list of adapter names, one per batch slot

_orig_lora_forward = _lora_layer.Linear.forward

def _patched_lora_forward(self, x, *args, **kwargs):
    adapters = GBMM_STEP_ADAPTERS
    if adapters is not None and not self.disable_adapters and all(a in self.lora_A for a in adapters):
        # Stack per-request A, B for this step: (N, R, H) and (N, O, R)
        A_stack = torch.stack([self.lora_A[a].weight for a in adapters], dim=0)
        B_stack = torch.stack([self.lora_B[a].weight for a in adapters], dim=0)
        scaling = self.scaling[adapters[0]]
        result = self.base_layer(x, *args, **kwargs)
        orig_dtype = result.dtype
        x_lora = x.to(A_stack.dtype)
        mid = torch.bmm(x_lora, A_stack.transpose(1, 2))    # (N, L, R)
        lora_out = torch.bmm(mid, B_stack.transpose(1, 2))  # (N, L, O)
        return (result + lora_out * scaling).to(orig_dtype)
    return _orig_lora_forward(self, x, *args, **kwargs)

_lora_layer.Linear.forward = _patched_lora_forward

# Two inference requests share the prompt and the infer adapter; gather-BMM
# batches them into a single (2, L) forward.
prompt = "Explain what machine learning is in one sentence."
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device).repeat(2, 1)

max_new_tokens = 128
warmup_steps = 3

# Pre-tokenize training samples from Alpaca, seq_len=300
dataset = load_dataset("tatsu-lab/alpaca", split="train")
train_all_ids = []
for i in range(max_new_tokens):
    s = dataset[i]
    text = f"### Instruction:\n{s['instruction']}\n\n### Input:\n{s['input']}\n\n### Response:\n{s['output']}"
    ids = tokenizer(text, return_tensors="pt", max_length=300, padding="max_length", truncation=True).input_ids
    train_all_ids.append(ids)
train_all_ids = torch.cat(train_all_ids, dim=0).to(model.device)

optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
generated_ids = input_ids.clone()  # (2, L)

decode_times = []
train_fwd_times = []
bwd_opt_times = []

# Prefill: batched (2, L) forward, per-request adapters stacked on the fly
GBMM_STEP_ADAPTERS = ["infer", "infer"]  # 2 requests this step, both use infer
model.eval()
with torch.no_grad():
    out = model(input_ids=input_ids, use_cache=True)
    past_key_values = out.past_key_values
    next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (2, 1)
GBMM_STEP_ADAPTERS = None
generated_ids = torch.cat([generated_ids, next_token], dim=-1)
next_pos = input_ids.shape[-1]

# Coserving loop: ONE batched decode (2 requests) + LoRA training step per iter
for step in range(max_new_tokens):
    if step == 0:
        # Step 0: prefill already decoded; train only (skip decode timing)
        if (next_token == tokenizer.eos_token_id).all().item():
            break
        t0 = t1 = 0.0  # no decode this step
    else:
        # --- Decode: one batched forward covers both inference requests ---
        cur_token = generated_ids[:, -1:]  # (2, 1)
        position_ids = torch.tensor([[next_pos]], device=model.device).expand(2, 1)
        # Per-step adapter assignment (could vary per step in real serving)
        step_adapters = ["infer", "infer"]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        GBMM_STEP_ADAPTERS = step_adapters
        model.eval()
        with torch.no_grad():
            out = model(input_ids=cur_token, past_key_values=past_key_values,
                        position_ids=position_ids, use_cache=True)
            past_key_values = out.past_key_values
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (2, 1)
        GBMM_STEP_ADAPTERS = None
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        generated_ids = torch.cat([generated_ids, next_token], dim=-1)
        next_pos += 1
        if (next_token == tokenizer.eos_token_id).all().item():
            break

    # --- PEFT forward (training, using ft adapter) ---
    train_ids = train_all_ids[step:step+1]
    model.set_adapter("ft")
    model.train()
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    loss = model(input_ids=train_ids, labels=train_ids).loss
    torch.cuda.synchronize()
    t3 = time.perf_counter()

    # --- Backward + optimizer ---
    torch.cuda.synchronize()
    t4 = time.perf_counter()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    torch.cuda.synchronize()
    t5 = time.perf_counter()

    if step >= warmup_steps:
        if step > 0:
            decode_times.append(t1 - t0)
        train_fwd_times.append(t3 - t2)
        bwd_opt_times.append(t5 - t4)

    if step == 30:
        ckpt = {k: v.data.clone().cpu() for k, v in model.named_parameters() if v.requires_grad}
        torch.save(ckpt, "weights_v0_fa_step30.pt")
        print(f"[CHECKPOINT] step 30 weight sum: {sum(v.sum().item() for v in ckpt.values()):.10f}")

    if step % 10 == 0:
        dt_decode = (t1 - t0) if step > 0 else 0.0
        print(f"step {step:3d} | loss={loss.item():.4f} | decode={dt_decode:.4f}s | train_fwd={t3-t2:.4f}s | bwd+opt={t5-t4:.4f}s")

response_1 = tokenizer.decode(generated_ids[0][input_ids.shape[-1]:], skip_special_tokens=True)
response_2 = tokenizer.decode(generated_ids[1][input_ids.shape[-1]:], skip_special_tokens=True)
print(f"\n--- Generated response (request 1) ---\n{response_1}")
print(f"\n--- Generated response (request 2) ---\n{response_2}")

if decode_times:
    avg_decode = sum(decode_times) / len(decode_times)
    avg_train_fwd = sum(train_fwd_times) / len(train_fwd_times)
    avg_bwd = sum(bwd_opt_times) / len(bwd_opt_times)
    print(f"\n--- Timing (after {warmup_steps} warmup steps, {len(decode_times)} measured steps) ---")
    print(f"  avg decode:             {avg_decode*1000:.2f} ms")
    print(f"  avg train_fwd:          {avg_train_fwd*1000:.2f} ms")
    print(f"  avg backward+optimizer: {avg_bwd*1000:.2f} ms")
    print(f"  avg total:              {(avg_decode+avg_train_fwd+avg_bwd)*1000:.2f} ms")
