"""Naive co-serving with torch.compile piecewise CUDA graph.

Same as hf_coserving_v0.py (alternating decode + LoRA train, 3 adapters) but uses:
  - torch.compile(mode="reduce-overhead") with piecewise CUDA graph
  - KV cache (DynamicCache) for decode path (1 token/step instead of full seq)
  - Prefill (step 0 decode) runs uncompiled, steps 1+ compiled
  - Gather-BMM: 2 inference requests (same infer adapter) share ONE batched
    decode forward via stacked-LoRA BMM; training uses ft adapter via normal PEFT
"""
import time
import torch
torch._dynamo.config.cache_size_limit = 64

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaAttention
import transformers.models.llama.modeling_llama as _llama_mod
from transformers.masking_utils import create_causal_mask
from peft import LoraConfig, get_peft_model
import peft.tuners.lora.layer as _lora_layer

torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Piecewise CUDA graph: attention + mask creation run eagerly
LlamaAttention.forward = torch.compiler.disable(LlamaAttention.forward)
_llama_mod.create_causal_mask = torch.compiler.disable(create_causal_mask)

model_name = "meta-llama/Llama-3.2-3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_name, dtype=torch.bfloat16, device_map="auto",
    attn_implementation="flash_attention_2",
)

# 2 adapters: ft (train) + infer (shared by both inference requests)
lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.0)
model = get_peft_model(model, lora_config, adapter_name="ft")
model.add_adapter("infer", lora_config)
model.set_adapter("ft")
model.print_trainable_parameters()

# ─────────────────────────────────────────────────────────────────────────
# Gather-BMM: per-request adapter assignment varies per step, so A/B are
# stacked ON THE FLY inside the patched LoRA forward using the current
# step's GBMM_STEP_ADAPTERS list. GBMM_STEP_ADAPTERS=None disables the
# gather-BMM path so training (ft) uses the normal PEFT forward.
# ─────────────────────────────────────────────────────────────────────────
GBMM_STEP_ADAPTERS = None

_orig_lora_forward = _lora_layer.Linear.forward

def _patched_lora_forward(self, x, *args, **kwargs):
    adapters = GBMM_STEP_ADAPTERS
    if adapters is not None and not self.disable_adapters and all(a in self.lora_A for a in adapters):
        A_stack = torch.stack([self.lora_A[a].weight for a in adapters], dim=0)
        B_stack = torch.stack([self.lora_B[a].weight for a in adapters], dim=0)
        scaling = self.scaling[adapters[0]]
        result = self.base_layer(x, *args, **kwargs)
        orig_dtype = result.dtype
        x_lora = x.to(A_stack.dtype)
        mid = torch.bmm(x_lora, A_stack.transpose(1, 2))
        lora_out = torch.bmm(mid, B_stack.transpose(1, 2))
        return (result + lora_out * scaling).to(orig_dtype)
    return _orig_lora_forward(self, x, *args, **kwargs)

_lora_layer.Linear.forward = _patched_lora_forward

prompt = "Explain what machine learning is in one sentence."
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device).repeat(2, 1)

max_new_tokens = 128
warmup_steps = 5
device = model.device
prompt_len = input_ids.shape[1]

# Pre-tokenize training samples from Alpaca, seq_len=300
dataset = load_dataset("tatsu-lab/alpaca", split="train")
train_all_ids = []
for i in range(max_new_tokens):
    s = dataset[i]
    text = f"### Instruction:\n{s['instruction']}\n\n### Input:\n{s['input']}\n\n### Response:\n{s['output']}"
    ids = tokenizer(text, return_tensors="pt", max_length=300, padding="max_length", truncation=True).input_ids
    train_all_ids.append(ids)
train_all_ids = torch.cat(train_all_ids, dim=0).to(device)

optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

# --------------------------------------------------------------------------
# Prefill (uncompiled, batched 2, gather-BMM)
# --------------------------------------------------------------------------
print("[prefill] Running uncompiled batched prefill (gather-BMM, batch=2)...")
GBMM_STEP_ADAPTERS = ["infer", "infer"]
model.eval()
with torch.no_grad():
    outputs = model(input_ids=input_ids, use_cache=True)
GBMM_STEP_ADAPTERS = None
past_kv = outputs.past_key_values
next_token = outputs.logits[:, -1:, :].argmax(dim=-1)  # (2, 1)
generated_tokens = [next_token]
print(f"[prefill] Done. Cache has {past_kv.get_seq_length()} tokens (batch=2).")

# --------------------------------------------------------------------------
# Compile
# --------------------------------------------------------------------------
model_compiled = torch.compile(model, mode="reduce-overhead")
print("[compile] reduce-overhead, piecewise (attention + mask disabled)")

# --------------------------------------------------------------------------
# Co-serving loop: step 0 = train only (prefill already decoded),
#                  steps 1+ = batched decode (gather-BMM) + train
# --------------------------------------------------------------------------
decode_times = []
train_fwd_times = []
bwd_opt_times = []

for step in range(max_new_tokens):
    if step > 0:
        # --- Batched decode: one forward covers both inference requests ---
        pos = prompt_len + step - 1
        position_ids = torch.tensor([[pos]], device=device).expand(2, 1)
        # Per-step adapter assignment (could vary per step in real serving)
        step_adapters = ["infer", "infer"]

        model_compiled.eval()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        GBMM_STEP_ADAPTERS = step_adapters
        with torch.no_grad():
            outputs = model_compiled(
                input_ids=next_token,
                position_ids=position_ids,
                past_key_values=past_kv,
                use_cache=True,
            )
            next_token = outputs.logits[:, -1:, :].argmax(dim=-1)  # (2, 1)
        GBMM_STEP_ADAPTERS = None
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        generated_tokens.append(next_token)

        if (next_token == tokenizer.eos_token_id).all().item():
            break
    else:
        t0 = t1 = time.perf_counter()  # no decode at step 0

    # --- Train forward (ft adapter, compiled, GBMM off) ---
    train_ids = train_all_ids[step:step+1]
    model_compiled.set_adapter("ft")
    model_compiled.train()
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    loss = model_compiled(input_ids=train_ids, labels=train_ids, use_cache=False).loss
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
        decode_times.append(t1 - t0)
        train_fwd_times.append(t3 - t2)
        bwd_opt_times.append(t5 - t4)

    if step == 30:
        ckpt = {k: v.data.clone().cpu() for k, v in model.named_parameters() if v.requires_grad}
        torch.save(ckpt, "weights_v0_compile_fa_step30.pt")
        print(f"[CHECKPOINT] step 30 weight sum: {sum(v.sum().item() for v in ckpt.values()):.10f}")

    if step % 10 == 0:
        print(f"step {step:3d} | loss={loss.item():.4f} | decode={t1-t0:.4f}s | train_fwd={t3-t2:.4f}s | bwd+opt={t5-t4:.4f}s")

# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------
all_tokens = torch.cat([input_ids] + generated_tokens, dim=-1)
response_1 = tokenizer.decode(all_tokens[0][prompt_len:], skip_special_tokens=True)
response_2 = tokenizer.decode(all_tokens[1][prompt_len:], skip_special_tokens=True)
print(f"\n--- Generated response (request 1) ---\n{response_1}")
print(f"\n--- Generated response (request 2) ---\n{response_2}")

if decode_times:
    avg_decode = sum(decode_times) / len(decode_times)
    avg_fwd = sum(train_fwd_times) / len(train_fwd_times)
    avg_bwd = sum(bwd_opt_times) / len(bwd_opt_times)
    print(f"\n--- Timing (after {warmup_steps} warmup, {len(decode_times)} measured steps) ---")
    print(f"  avg decode (KV-cache):  {avg_decode*1000:.2f} ms")
    print(f"  avg train_fwd:          {avg_fwd*1000:.2f} ms")
    print(f"  avg backward+optimizer: {avg_bwd*1000:.2f} ms")
    print(f"  avg total:              {(avg_decode+avg_fwd+avg_bwd)*1000:.2f} ms")
