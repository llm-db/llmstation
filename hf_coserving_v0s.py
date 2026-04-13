"""
Co-serving v0s: multiprocessing (spawn) with shared GPU memory via CUDA IPC.
Schedule: decode || train_fwd (parallel) -> backward+optimizer (sequential)

The decode worker receives model parameters/buffers as CUDA IPC handles, so
both processes access the SAME physical GPU memory. No model weight duplication.

Worker startup: builds model architecture on CPU (from HuggingFace cache),
then replaces all parameters and buffers with the shared GPU tensors.

Safety: decode reads base + infer LoRA weights. Forward reads base + ft LoRA
weights, writes activations (no param mutation). Backward reads base + ft LoRA
weights, writes ft .grad. Optimizer writes ft .data only.
Decode || forward: both are read-only on base weights, safe to overlap.
"""

import time
import torch
import torch.multiprocessing as mp
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

MODEL_NAME = "meta-llama/Llama-3.2-3B"
MAX_NEW_TOKENS = 128
WARMUP_STEPS = 3
SEQ_LEN = 300


def decode_worker(shared_state_q, task_q, result_q):
    """Decode worker. Builds model architecture, then uses shared GPU tensors."""
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Receive shared GPU tensors via CUDA IPC (opens handles to parent's GPU memory)
    shared_state = shared_state_q.get()
    n_shared = len(shared_state)
    print(f"[decode_worker] Received {n_shared} shared GPU tensors via CUDA IPC")

    # Build model architecture (weights load to CPU, then get replaced with GPU)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.0)
    model = get_peft_model(model, lora_config, adapter_name="ft")
    model.add_adapter("infer", lora_config)

    # Replace all params and buffers with shared GPU tensors (same physical GPU memory)
    replaced = 0
    for name, param in model.named_parameters():
        if name in shared_state:
            param.data = shared_state[name]
            replaced += 1
    for name, buf in model.named_buffers():
        if name in shared_state:
            buf.data = shared_state[name]
            replaced += 1
    print(f"[decode_worker] Replaced {replaced}/{n_shared} tensors with shared GPU memory")

    model.set_adapter("infer")
    model.eval()

    result_q.put("ready")

    while True:
        msg = task_q.get()
        if msg is None:
            break
        generated_ids = msg.cuda()

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            logits = model(input_ids=generated_ids).logits[:, -1, :]
            next_token = logits.argmax(dim=-1, keepdim=True)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        result_q.put((next_token.cpu(), t1 - t0))


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    model_name = MODEL_NAME
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
    model.set_adapter("ft")
    model.print_trainable_parameters()

    # Inference prompt
    prompt = "Explain what machine learning is in one sentence."
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

    # Pre-tokenize training samples from Alpaca, seq_len=300
    dataset = load_dataset("tatsu-lab/alpaca", split="train")
    train_all_ids = []
    for i in range(MAX_NEW_TOKENS):
        s = dataset[i]
        text = f"### Instruction:\n{s['instruction']}\n\n### Input:\n{s['input']}\n\n### Response:\n{s['output']}"
        ids = tokenizer(text, return_tensors="pt", max_length=SEQ_LEN, padding="max_length", truncation=True).input_ids
        train_all_ids.append(ids)
    train_all_ids = torch.cat(train_all_ids, dim=0).to(model.device)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    generated_ids = input_ids.clone()

    # Collect shared state: all params + buffers as GPU tensors
    # When sent through Queue, CUDA tensors are serialized as IPC handles
    shared_state = {}
    for name, param in model.named_parameters():
        shared_state[name] = param.data
    for name, buf in model.named_buffers():
        shared_state[name] = buf.data

    # Spawn decode worker with shared GPU memory
    shared_state_q = mp.Queue()
    task_q = mp.Queue()
    result_q = mp.Queue()

    print(f"Spawning decode worker with {len(shared_state)} shared GPU tensors via CUDA IPC...")
    decode_proc = mp.Process(target=decode_worker, args=(shared_state_q, task_q, result_q))
    decode_proc.start()
    shared_state_q.put(shared_state)

    assert result_q.get() == "ready"
    print("Decode worker ready (shared GPU memory, single model copy).\n")

    # ── Co-serving loop ───────────────────────────────────────────────────

    parallel_times = []
    decode_times = []
    train_fwd_times = []
    bwd_opt_times = []

    for step in range(MAX_NEW_TOKENS):
        # ==============================================================
        # Phase 1: Decode || Train forward (parallel)
        #   - Decode worker: infer adapter forward (shared GPU memory)
        #   - Main process: ft adapter forward (compute loss)
        # ==============================================================
        model.set_adapter("ft")
        model.train()
        train_ids = train_all_ids[step:step+1]

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        # Dispatch decode to worker (non-blocking)
        task_q.put(generated_ids.cpu())

        # Train forward in main process (runs in parallel with decode worker)
        loss = model(input_ids=train_ids, labels=train_ids).loss
        torch.cuda.synchronize()
        t1_fwd = time.perf_counter()

        # Wait for decode result
        next_token, decode_time = result_q.get()
        t1 = time.perf_counter()

        generated_ids = torch.cat([generated_ids, next_token.to(model.device)], dim=-1)

        if next_token.item() == tokenizer.eos_token_id:
            break

        # ==============================================================
        # Phase 2: Backward + Optimizer (sequential, main process only)
        # ==============================================================
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.synchronize()
        t3 = time.perf_counter()

        train_fwd_time = t1_fwd - t0
        parallel_time = t1 - t0
        bwd_time = t3 - t2

        if step >= WARMUP_STEPS:
            parallel_times.append(parallel_time)
            decode_times.append(decode_time)
            train_fwd_times.append(train_fwd_time)
            bwd_opt_times.append(bwd_time)

        if step == 30:
            ckpt = {k: v.data.clone().cpu() for k, v in model.named_parameters() if v.requires_grad}
            torch.save(ckpt, "weights_v0s_fa_step30.pt")
            print(f"[CHECKPOINT] step 30 weight sum: {sum(v.sum().item() for v in ckpt.values()):.10f}")

        if step % 10 == 0:
            total = parallel_time + bwd_time
            print(f"step {step:3d} | loss={loss.item():.4f} | total={total:.4f}s | "
                  f"parallel={parallel_time:.4f}s (decode={decode_time:.4f}s, train_fwd={train_fwd_time:.4f}s) | "
                  f"bwd+opt={bwd_time:.4f}s")

    # ── Shutdown ──────────────────────────────────────────────────────────

    task_q.put(None)
    decode_proc.join()

    response = tokenizer.decode(generated_ids[0][input_ids.shape[-1]:], skip_special_tokens=True)
    print(f"\n--- Generated response ---\n{response}")

    if parallel_times:
        avg_par = sum(parallel_times) / len(parallel_times)
        avg_decode = sum(decode_times) / len(decode_times)
        avg_fwd = sum(train_fwd_times) / len(train_fwd_times)
        avg_bwd = sum(bwd_opt_times) / len(bwd_opt_times)
        avg_total = avg_par + avg_bwd

        print(f"\n--- Timing (after {WARMUP_STEPS} warmup steps, {len(parallel_times)} measured steps) ---")
        print(f"  Phase 1 (parallel, decode || train_fwd):")
        print(f"    avg parallel wall:      {avg_par*1000:.2f} ms")
        print(f"    avg decode (worker):    {avg_decode*1000:.2f} ms")
        print(f"    avg train_fwd (main):   {avg_fwd*1000:.2f} ms")
        print(f"  Phase 2 (sequential):")
        print(f"    avg bwd+opt:            {avg_bwd*1000:.2f} ms")
        print(f"  Total:")
        print(f"    avg total:              {avg_total*1000:.2f} ms")

        avg_v0_equiv = avg_decode + avg_fwd + avg_bwd
        overlap = avg_v0_equiv - avg_total
        print(f"\n  --- vs v0 sequential (decode + train_fwd + bwd) ---")
        print(f"  v0 equivalent:            {avg_v0_equiv*1000:.2f} ms")
        if overlap > 0:
            print(f"  overlap saved:            {overlap*1000:.2f} ms ({overlap/avg_v0_equiv*100:.1f}%)")
        else:
            print(f"  overhead:                 {-overlap*1000:.2f} ms ({-overlap/avg_v0_equiv*100:.1f}% extra)")
