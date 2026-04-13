import time
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

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

# Apply two LoRA adapters: one for inference, one for training
lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.0)
model = get_peft_model(model, lora_config, adapter_name="ft")
model.add_adapter("infer", lora_config)
model.set_adapter("ft")  # only ft adapter requires grad
model.print_trainable_parameters()

# Inference prompt
prompt = "Explain what machine learning is in one sentence."
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

max_new_tokens = 128
warmup_steps = 3

# Pre-tokenize training samples from Alpaca, seq_len=200
dataset = load_dataset("tatsu-lab/alpaca", split="train")
train_all_ids = []
for i in range(max_new_tokens):
    s = dataset[i]
    text = f"### Instruction:\n{s['instruction']}\n\n### Input:\n{s['input']}\n\n### Response:\n{s['output']}"
    ids = tokenizer(text, return_tensors="pt", max_length=300, padding="max_length", truncation=True).input_ids
    train_all_ids.append(ids)
train_all_ids = torch.cat(train_all_ids, dim=0).to(model.device)

optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
generated_ids = input_ids.clone()

decode_times = []
train_fwd_times = []
bwd_opt_times = []

# Coserving loop: alternate decode and LoRA steps
for step in range(max_new_tokens):
    # --- Decode (inference forward, using infer adapter) ---
    model.set_adapter("infer")
    model.eval()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(input_ids=generated_ids).logits[:, -1, :]
        next_token = logits.argmax(dim=-1, keepdim=True)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    generated_ids = torch.cat([generated_ids, next_token], dim=-1)
    if next_token.item() == tokenizer.eos_token_id:
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
        decode_times.append(t1 - t0)
        train_fwd_times.append(t3 - t2)
        bwd_opt_times.append(t5 - t4)

    if step == 30:
        ckpt = {k: v.data.clone().cpu() for k, v in model.named_parameters() if v.requires_grad}
        torch.save(ckpt, "weights_v0_fa_step30.pt")
        print(f"[CHECKPOINT] step 30 weight sum: {sum(v.sum().item() for v in ckpt.values()):.10f}")

    if step % 10 == 0:
        print(f"step {step:3d} | loss={loss.item():.4f} | decode={t1-t0:.4f}s | train_fwd={t3-t2:.4f}s | bwd+opt={t5-t4:.4f}s")

response = tokenizer.decode(generated_ids[0][input_ids.shape[-1]:], skip_special_tokens=True)
print(f"\n--- Generated response ---\n{response}")

if decode_times:
    avg_decode = sum(decode_times) / len(decode_times)
    avg_train_fwd = sum(train_fwd_times) / len(train_fwd_times)
    avg_bwd = sum(bwd_opt_times) / len(bwd_opt_times)
    print(f"\n--- Timing (after {warmup_steps} warmup steps, {len(decode_times)} measured steps) ---")
    print(f"  avg decode:             {avg_decode*1000:.2f} ms")
    print(f"  avg train_fwd:          {avg_train_fwd*1000:.2f} ms")
    print(f"  avg backward+optimizer: {avg_bwd*1000:.2f} ms")
    print(f"  avg total:              {(avg_decode+avg_train_fwd+avg_bwd)*1000:.2f} ms")
