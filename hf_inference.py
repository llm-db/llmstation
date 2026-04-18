import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "meta-llama/Llama-3.1-8B"
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

# Prefill: process the full prompt and build the KV cache
with torch.no_grad():
    out = model(input_ids=input_ids, use_cache=True)
    past_key_values = out.past_key_values
    logits = out.logits[:, -1, :]
    next_token = logits.argmax(dim=-1, keepdim=True)
generated_ids = torch.cat([generated_ids, next_token], dim=-1)
next_pos = input_ids.shape[-1]  # position of the token we just generated

for step in range(max_new_tokens):
    if step == 0:
        # Step 0 was the prefill decode above; skip timing but check EOS
        if next_token.item() == tokenizer.eos_token_id:
            break
        continue

    # Decode with KV cache: feed only the last token + position_ids
    cur_token = generated_ids[:, -1:]
    position_ids = torch.tensor([[next_pos]], device=model.device)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(input_ids=cur_token, past_key_values=past_key_values,
                    position_ids=position_ids, use_cache=True)
        past_key_values = out.past_key_values
        logits = out.logits[:, -1, :]
        next_token = logits.argmax(dim=-1, keepdim=True)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    generated_ids = torch.cat([generated_ids, next_token], dim=-1)
    next_pos += 1
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
