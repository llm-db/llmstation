"""
Co-serving v0s: multiprocessing (spawn) with shared GPU memory via CUDA IPC.
Schedule: decode || train_fwd (parallel) -> backward+optimizer (sequential)

Decode batches N inference requests into ONE forward via gather-BMM:
shared base-weight matmul, per-sample LoRA applied with stacked A/B weights
and torch.bmm. Training uses a second adapter (ft).

The per-request adapter list is sent with each decode task (real serving has
variable batch size / adapter mix per step), and the worker's patched PEFT
LoraLinear stacks A/B weights on the fly based on that list.

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
    """Batched decode worker: 2 inference requests fused via gather-BMM."""
    import peft.tuners.lora.layer as _lora_layer

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

    # Gather-BMM: per-step adapter assignment varies across steps (real serving
    # has different request counts and adapter mixes), so A/B are stacked ON
    # THE FLY inside the patched forward using step_adapters_ref[0]. The main
    # process supplies the per-step list with each decode task.
    step_adapters_ref = [None]  # closure-accessible mutable holder

    _orig_lora_forward = _lora_layer.Linear.forward

    def _patched_lora_forward(self, x, *args, **kwargs):
        adapters = step_adapters_ref[0]
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

    model.eval()

    past_key_values = None

    result_q.put("ready")

    while True:
        msg = task_q.get()
        if msg is None:
            break

        op = msg[0]

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            if op == "prefill":
                # Prefill: batched (N, L_prompt) forward builds KV cache of batch N
                input_ids = msg[1].cuda()
                step_adapters_ref[0] = list(msg[2])
                out = model(input_ids=input_ids, use_cache=True)
                past_key_values = out.past_key_values
                logits = out.logits[:, -1, :]
            else:
                # Decode: batched (N, 1) token with KV cache
                cur_token = msg[1].cuda()
                kv_pos = msg[2]
                step_adapters_ref[0] = list(msg[3])
                n = len(step_adapters_ref[0])
                position_ids = torch.tensor([[kv_pos]], device=cur_token.device).expand(n, 1)
                out = model(input_ids=cur_token, past_key_values=past_key_values,
                            position_ids=position_ids, use_cache=True)
                past_key_values = out.past_key_values
                logits = out.logits[:, -1, :]
            next_token = logits.argmax(dim=-1, keepdim=True)  # (N, 1)
            step_adapters_ref[0] = None

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

    # 2 LoRA adapters: ft (train) + infer (shared by both inference requests)
    lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.0)
    model = get_peft_model(model, lora_config, adapter_name="ft")
    model.add_adapter("infer", lora_config)
    model.set_adapter("ft")
    model.print_trainable_parameters()

    # Two inference requests share the prompt and the infer adapter; gather-BMM batches them
    prompt = "Explain what machine learning is in one sentence."
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device).repeat(2, 1)

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
    generated_ids = input_ids.clone()  # (2, L)

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

    generated_tokens = []  # list of (2, 1) CPU tensors, batched 2 requests
    # Position of the last token consumed by the KV cache (same for both samples).
    kv_pos = input_ids.shape[-1] - 1

    for step in range(MAX_NEW_TOKENS):
        # ==============================================================
        # Phase 1: Decode (batched 2 via gather-BMM) || Train forward
        #   - Decode worker: single batched forward covers both requests
        #   - Main process: ft adapter forward (compute loss)
        # ==============================================================
        model.set_adapter("ft")
        model.train()
        train_ids = train_all_ids[step:step+1]

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        # Per-step adapter assignment (could vary per step in real serving)
        step_adapters = ["infer", "infer"]

        # Dispatch decode to worker (non-blocking)
        if step == 0:
            # Prefill: send full batched prompt + adapter list
            task_q.put(("prefill", input_ids.cpu(), step_adapters))
            kv_pos = input_ids.shape[-1] - 1
        else:
            # Decode: send last generated batched token + position + adapter list
            last_token = generated_tokens[-1]  # CPU (N, 1)
            kv_pos += 1
            task_q.put(("decode", last_token, kv_pos, step_adapters))

        # Train forward in main process (runs in parallel with decode worker)
        loss = model(input_ids=train_ids, labels=train_ids).loss
        torch.cuda.synchronize()
        t1_fwd = time.perf_counter()

        # Wait for decode result
        next_token, decode_time = result_q.get()  # (2, 1) CPU
        t1 = time.perf_counter()

        generated_tokens.append(next_token)
        generated_ids = torch.cat([generated_ids, next_token.to(model.device)], dim=-1)

        if (next_token == tokenizer.eos_token_id).all().item():
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

    response_1 = tokenizer.decode(generated_ids[0][input_ids.shape[-1]:], skip_special_tokens=True)
    response_2 = tokenizer.decode(generated_ids[1][input_ids.shape[-1]:], skip_special_tokens=True)
    print(f"\n--- Generated response (request 1) ---\n{response_1}")
    print(f"\n--- Generated response (request 2) ---\n{response_2}")

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
