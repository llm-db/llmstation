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

# Apply LoRA
lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.0)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

max_steps = 128

# Pre-tokenize training samples from Alpaca, seq_len=200
dataset = load_dataset("tatsu-lab/alpaca", split="train")
train_all_ids = []
for i in range(max_steps):
    s = dataset[i]
    text = f"### Instruction:\n{s['instruction']}\n\n### Input:\n{s['input']}\n\n### Response:\n{s['output']}"
    ids = tokenizer(text, return_tensors="pt", max_length=300, padding="max_length", truncation=True).input_ids
    train_all_ids.append(ids)
train_all_ids = torch.cat(train_all_ids, dim=0).to(model.device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
warmup_steps = 3

fwd_times = []
bwd_opt_times = []

model.train()
for step in range(max_steps):
    # --- Forward ---
    train_ids = train_all_ids[step:step+1]
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    loss = model(input_ids=train_ids, labels=train_ids).loss
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    # --- Backward + optimizer ---
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    torch.cuda.synchronize()
    t3 = time.perf_counter()

    if step >= warmup_steps:
        fwd_times.append(t1 - t0)
        bwd_opt_times.append(t3 - t2)

    if step == 30:
        ckpt = {k: v.data.clone().cpu() for k, v in model.named_parameters() if v.requires_grad}
        torch.save(ckpt, "weights_peft_fa_step30.pt")
        print(f"[CHECKPOINT] step 30 weight sum: {sum(v.sum().item() for v in ckpt.values()):.10f}")

    if step % 10 == 0:
        print(f"step {step:3d} | loss={loss.item():.4f} | fwd={t1-t0:.4f}s | bwd+opt={t3-t2:.4f}s")

if fwd_times:
    avg_fwd = sum(fwd_times) / len(fwd_times)
    avg_bwd = sum(bwd_opt_times) / len(bwd_opt_times)
    print(f"\n--- Timing (after {warmup_steps} warmup steps, {len(fwd_times)} measured steps) ---")
    print(f"  avg fwd:                {avg_fwd*1000:.2f} ms")
    print(f"  avg backward+optimizer: {avg_bwd*1000:.2f} ms")
    print(f"  avg total:              {(avg_fwd+avg_bwd)*1000:.2f} ms")
