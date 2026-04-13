import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "meta-llama/Llama-3.2-3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_name, dtype=torch.bfloat16, device_map="auto",
    attn_implementation="flash_attention_2",
)

# Inference prompt
prompt = "Explain what machine learning is in one sentence."
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

generated_ids = input_ids.clone()
max_new_tokens = 128
warmup_steps = 3

decode_times = []

model.eval()
for step in range(max_new_tokens):
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

    if step >= warmup_steps:
        decode_times.append(t1 - t0)

    if step % 10 == 0:
        print(f"step {step:3d} | decode={t1-t0:.4f}s")

response = tokenizer.decode(generated_ids[0][input_ids.shape[-1]:], skip_special_tokens=True)
print(f"\n--- Generated response ---\n{response}")

if decode_times:
    avg_decode = sum(decode_times) / len(decode_times)
    print(f"\n--- Timing (after {warmup_steps} warmup steps, {len(decode_times)} measured steps) ---")
    print(f"  avg decode: {avg_decode*1000:.2f} ms")
