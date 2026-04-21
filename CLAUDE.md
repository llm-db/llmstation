## Project: LLMStation Fusion Engine

Benchmarking co-serving (simultaneous multi-request inference + LoRA fine-tuning) on a single GPU.

- **Model:** meta-llama/Llama-3.1-8B, bf16
- **GPU:** NVIDIA RTX 3090 (24GB, sm_86), 64-core CPU, 251GB RAM.
- **Conda env:** llmstation (Python 3.14.3, PyTorch 2.11.0+cu126)
- **Attention:** All files use Flash Attention 2 (flash-attn 2.8.3)

## Workload (all co-serving variants)

Per step:
- **2 inference requests** sharing the single "infer" LoRA adapter (batch dim 2 in decode)
- **1 LoRA training step** on the "ft" adapter (seq_len=300, Alpaca)
- Inference uses a KV cache; training has no cache
- Per-step adapter routing is decided dynamically (no pre-built adapter caches), mirroring real multi-tenant serving where the request count / adapter mix varies across steps

## Files

| File | Role |
|------|------|
| `hf_inference.py` | Pure inference benchmark (reference) |
| `hf_peft.py` | Pure LoRA training benchmark (reference, seq_len=300) |
| `hf_coserving_v0.py` | Naive co-serving: batched 2-request decode via **gather-BMM** (per-step dynamic adapter stacking monkey-patched into PEFT's LoraLinear) + separate PEFT training step |
| `hf_coserving_v0s.py` | v0 + multiprocessing (spawn + CUDA IPC): batched decode in worker ‖ train_fwd in main process, then bwd+opt sequentially |
| `hf_coserving_v1.py` | Matrix-level fusion: infer `(B_i, L_i, H)` flattened and concatenated with train `(1, L_t, H)` for a single base matmul; infer LoRA via **gather-BMM**, train LoRA via ft adapter |
| `hf_coserving_v2.py` | v1 + `FusedBaseLinear` custom autograd: train-only backward skips the `B_i*L_i` infer tokens |
| `hf_coserving_v3.py` | v2 fusion + **SGMV Triton kernels** (shrink + expand). 3 segments (2 infer + 1 train) → 2 unique adapter slots. Infer adapter read directly from PEFT `Parameter` storage (no stacking, no per-request duplication); only a minimal bf16 buffer for ft's fp32→bf16 dtype cast. Per-segment metadata passed as int32/fp32 tensors |
| `hf_coserving_v0_compile.py` | v0 + `torch.compile(mode="reduce-overhead")` piecewise CUDA graph + DynamicCache batch=2 |
| `hf_coserving_v3_compile.py` | v3 + torch.compile + KV cache; attn disabled from compile (SGMV + flash_attn run eagerly), MLP/norms/embed/lm_head compiled with static fused shape `(1, 2+300)` |
| `compare_weights.py` | Compares step-30 ft-adapter checkpoints across peft/v0/v0s/v0_compile/v1/v2/v3/v3_compile |
| `requirements.txt` | torch, transformers, accelerate, peft, datasets, ninja, flash-attn, einops, triton |

## LoRA config
- r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.0
- Two adapters on all co-serving files:
  - **"infer"** — shared by both inference requests (frozen, bf16)
  - **"ft"** — training adapter (trainable, stored fp32 for optimizer precision)

## Key results (128 steps, 2 inference requests + 1 training step/iter, seq_len=300)

**Speed (avg per step, ms):** Non-compile warmup=3, compile warmup=5.

| Version | decode / fused_fwd | train_fwd | bwd+opt | total | vs baseline |
|---------|--------------------|-----------|---------|-------|-------------|
| inference (1 req, KV cache) | 30.35 | — | — | 30.35 | (ref) |
| peft (train only) | — | 129.43 | 138.43 | 267.86 | (ref) |
| v0 (gather-BMM naive) | 36.61 | 130.07 | 139.38 | 306.06 | baseline |
| v0s (gather-BMM, decode‖train, +MPS) | parallel wall 142.46 | — | 138.38 | **280.84** | −8.2% |
| v1 (gather-BMM + fusion) | 129.70 | — | 141.17 | 270.87 | −11.5% |
| v2 (+ train-only backward) | 129.94 | — | 137.92 | 267.86 | −12.5% |
| v3 (SGMV, direct infer access) | 127.00 | — | 137.67 | **264.67** | −13.5% |
| v0_compile (gather-BMM + compile) | 31.66 | 125.28 | 129.16 | 286.10 | baseline_c |
| v3_compile (SGMV + compile) | 121.07 | — | 128.58 | **249.65** | −12.7% |

Notes:
- Baselines: **v0** (non-compile) and **v0_compile** (compile). peft is a training-only reference and is not treated as a baseline.
- In v0/v0s/v0_compile the decode is a standalone batched `(2, 1)` forward and train_fwd is a separate call; in v1/v2/v3/v3_compile both are concatenated into one fused matmul (`fused_fwd`).
- v0s + **CUDA MPS wins** on 3090 (−8.2% vs v0): MPS lets the two processes' kernels coexist on the SMs instead of time-slicing at the CUDA-context level, so decode‖train_fwd overlap is clean. Without MPS the same setup still wins on 3090 (−3.1%, 296.47 ms) thanks to 3090's 936 GB/s HBM keeping contention mild; on L4 (300 GB/s) without MPS it lost by +2.3%. MPS usage: start daemon with `CUDA_VISIBLE_DEVICES=2 nvidia-cuda-mps-control -d`, run python with `CUDA_VISIBLE_DEVICES=0` (daemon remaps its single visible GPU to index 0 for clients) — see Architecture notes for the setup recipe and pitfalls.
- v3 is the fastest non-compile variant (−13.5% vs v0); v3_compile is the fastest overall (−12.7% vs v0_compile).
- Relative gains are smaller than the 3B/L4 numbers in git history (v3 was −22.1%, v3_compile −22.4%). 8B's larger base matmul (hidden 4096, inter 14336) dominates each step, shrinking the relative impact of LoRA-side and backward-side optimizations.

**Weight equivalence:** not re-verified on 8B. `compare_weights.py` was last run against 3B/L4 checkpoints (all variants within atol=5e-3 of pure peft on 112 trainable LoRA params). After the 3B→8B swap, step-30 checkpoints need to be regenerated per variant and `compare_weights.py` re-run before the equivalence claim can be restated.

## Why each version is faster than the previous

### v0s: multiprocessing with shared GPU + CUDA MPS (−25.22 ms, −8.2% vs v0)
Spawn + CUDA IPC puts two processes on the same physical GPU memory and runs the batched 2-request decode in parallel with train_fwd. CUDA MPS (Multi-Process Service) lets both processes' kernels run concurrently on the SMs instead of time-slicing at the CUDA-context level, which is what actually turns the "parallel wall" into a meaningful saving. Per-op inflation under MPS is modest (decode 36.6 → 45.0 ms, ~1.23×; train_fwd 130.1 → 141.4 ms, ~1.09×), and the sequential sum 36.6+130.1=166.7 ms collapses to parallel wall 142.5 ms — an overlap saving of ~44 ms (13.5% of v0-equivalent cost). Without MPS the same setup saves only ~9 ms (inflation climbs to ~1.36× / 1.20×, parallel wall 157.4 ms) because the two processes context-switch at coarse granularity instead of sharing SMs. On L4 (300 GB/s) without MPS the setup lost (+2.3%) because lower HBM bandwidth amplified contention to ~1.5–2×; MPS on lower-bandwidth GPUs is worth re-measuring.

### v0 → v1: matrix-level fusion (−35.19 ms, −11.5% vs v0)
v0 runs decode and train as fully separate forward passes — every base weight matrix is read from HBM twice per layer. v1 flattens the inference batch `(2, L_i, H) → (1, 2*L_i, H)` and concatenates with the training tokens `(1, L_t, H)` along the sequence dim for a single base-weight matmul covering both. Halves HBM bandwidth on the dominant base-weight loads. Infer LoRA uses gather-BMM; train LoRA keeps the ft adapter via PEFT.

### v1 → v2: train-only backward (−3.01 ms, −1.1% vs v1)
v1's autograd backward still processes the full `(2*L_i + L_t)` dimension even though infer contributes zero gradient. `FusedBaseLinear(torch.autograd.Function)` keeps v1's fused forward but overrides backward to compute only `grad_train @ W` on the `L_t` training tokens, skipping the `2*L_i` infer tokens entirely. Savings are small here because L_t=300 dominates but would grow with the infer/train ratio.

### v2 → v3: SGMV with direct adapter access (−3.19 ms, −1.2% vs v2)
Two complementary properties of SGMV over gather-BMM:

1. **No per-request weight duplication.** Gather-BMM needs to stack the infer adapter weight as many times as there are inference requests (`(B_i, R, H)` for the A matrix). SGMV stores each unique adapter once, and N segments route to K ≤ N unique slots. With 2 infer requests sharing one "infer" adapter, K=2 total unique slots (infer + ft) — not 3.
2. **No extra materialized copy of the infer adapter at all.** The Triton kernel receives `proj.lora_A["infer"].weight.data_ptr()` directly. The only remaining copy is a tiny one-shape bf16 buffer for the ft adapter, needed because PEFT stores ft in fp32 and the kernel reads bf16 — this is a dtype-conversion buffer, not a duplicate for batching.

Each shrink / expand kernel takes two adapter pointers (`a_infer_ptr`, `a_ft_bf16_ptr`) and uses `tl.where(adapter_id == 0, a_infer_tile, a_ft_tile)` at block granularity to route. Still 2 Triton launches per LoRA projection regardless of request count. Forward drops ~2.9 ms; backward drops ~0.25 ms (bf16 LoRA grads with recomputed `train_mid`, matching v2 cost).

### v0 → v0_compile: compile applied to the non-compile baseline (−19.96 ms, −6.5% vs v0)
Same batched-decode + separate-train structure as v0, but with `torch.compile(mode="reduce-overhead")` + DynamicCache. Attention and mask creation run eagerly, everything else gets CUDA-graphed. Sets the compile baseline at 286.10 ms.

### v3 → v3_compile: compile applied to the best non-compile variant (−15.02 ms, −5.7% vs v3; −12.7% vs v0_compile)
Attention (flash_attn + SGMV + KV cache mutation) stays eager via `@torch.compiler.disable`. Embed, layernorms, MLP, lm_head all compile into CUDA graphs with static fused shape `(1, 2 + 300) = (1, 302)` and static decode input `(2, 1)`. Segment tensors are built in the main loop (outside the compiled region) so Dynamo never sees `torch.tensor([...])`; kernels execute in the disabled attn path.

## Architecture notes

### Gather-BMM (v0 / v0s / v0_compile / v1 / v2)
Batched per-request LoRA for N inference slots:
- For v0 / v0s / v0_compile: monkey-patch `peft.tuners.lora.layer.Linear.forward`. For v1 / v2: call directly inside the custom `fused_linear`.
- Before each decode forward, set `GBMM_STEP_ADAPTERS = [...]` (one adapter name per slot; varies per step).
- Inside the patched / custom forward, stack on the fly:
  - `A_stack = torch.stack([proj.lora_A[a].weight for a in adapters], dim=0)` → `(N, R, H)`
  - `B_stack = torch.stack([proj.lora_B[a].weight for a in adapters], dim=0)` → `(N, O, R)`
- Per-sample BMM: `mid = bmm(x, A_stack.transpose(1,2))`; `lora = bmm(mid, B_stack.transpose(1,2)) * scaling`.
- Limitation: when M slots share the same adapter, its weight is materialized M times in the stack.

### SGMV direct (v3 / v3_compile)
Segment-based routing with direct adapter pointers:
- Per forward, build 4 tensors: `seg_starts`, `seg_lens`, `seg_adapters`, `seg_scalings` (one entry per segment, on GPU). For the current workload that's 3 entries (2 infer segments pointing at adapter 0, 1 train segment pointing at adapter 1).
- Infer adapter: `proj.lora_A["infer"].weight` / `proj.lora_B["infer"].weight` passed to the kernel as-is (bf16, no copy).
- Ft adapter: tiny `(R, H)` / `(O, R)` bf16 buffer per shape in `_ft_buf_cache`, updated per forward via `.copy_(ft.weight)` to convert fp32 → bf16.
- Triton kernels take two adapter pointers; `tl.where(adapter_id == 0, a_infer, a_ft)` selects per block.
- Shrink grid: `(cdiv(max_seg, BLOCK_M), n_segs)`. Expand grid: `(cdiv(max_seg, BLOCK_M), cdiv(H_out, BLOCK_N), n_segs)`. `n_segs` is passed at launch time so the structure scales with request count.
- `FusedLinearSGMV(torch.autograd.Function)`: forward = 1 cuBLAS (fused base matmul) + 2 Triton. Backward = bf16 LoRA gradient with recomputed `train_mid` (no large fp32 casts, 4 saved tensors).

### v1 / v2 fusion (matrix level, gather-BMM LoRA)
- `fused_linear(proj, infer_x, train_x)`:
  - `infer_flat = infer_x.reshape(1, B_i*L_i, H)` → cat with `train_x` along seq dim.
  - Single base matmul on `(1, B_i*L_i + L_t, H)`.
  - Gather-BMM LoRA for the infer segments, PEFT ft path for train.
  - Split back: `infer_base[:, :B_i*L_i].reshape(B_i, L_i, O)` / `train_base[:, B_i*L_i:]`.
- `attn_forward`: fused Q/K/V/O projections; separate RoPE + `flash_attn_func` for infer (batch `B_i`) and train (batch 1).
- `mlp_forward`: fused gate/up/down, SiLU activation separately.
- KV cache per layer: `(B_i, kv_len, num_kv_heads, head_dim)`; grows by `L_i` per step.
- v2 replaces the fused base matmul with `FusedBaseLinear(torch.autograd.Function)` whose backward only propagates through the train slice of the concatenated tensor.

### v0s multiprocessing (+ CUDA MPS)
- Spawn + CUDA IPC to share ~480 tensors (params + buffers) between processes — both reference the same physical GPU memory.
- Decode worker: builds the model on CPU, then rebinds every `param.data` / `buf.data` to the shared GPU tensor; also monkey-patches PEFT's LoraLinear for gather-BMM with the worker's `step_adapters_ref`.
- Per step: main dispatches `(op, batched_tokens, [kv_pos,] step_adapters)` to worker (non-blocking), runs train_fwd locally, waits for worker's batched next tokens, then runs bwd+opt.
- **CUDA MPS setup** (required to hit the 280.84 ms number — without it v0s is ~296 ms):
  - Daemon (pins to GPU 2, uses CUDA 12.6 to match torch/flash_attn):
    ```bash
    mkdir -p .mps/pipe .mps/log
    CUDA_VISIBLE_DEVICES=2 \
    CUDA_HOME=/usr/local/cuda-12.6 PATH=/usr/local/cuda-12.6/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH \
    CUDA_MPS_PIPE_DIRECTORY=$PWD/.mps/pipe CUDA_MPS_LOG_DIRECTORY=$PWD/.mps/log \
    nvidia-cuda-mps-control -d
    ```
  - Client (both main process and `decode_worker` — spawn inherits env):
    ```bash
    CUDA_VISIBLE_DEVICES=0 \
    CUDA_HOME=/usr/local/cuda-12.6 PATH=/usr/local/cuda-12.6/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH \
    CUDA_MPS_PIPE_DIRECTORY=$PWD/.mps/pipe CUDA_MPS_LOG_DIRECTORY=$PWD/.mps/log \
    python hf_coserving_v0s.py
    ```
  - Teardown: `CUDA_MPS_PIPE_DIRECTORY=$PWD/.mps/pipe echo quit | nvidia-cuda-mps-control`
  - **Daemon uses physical index (`=2`), client uses `=0`.** Daemon remaps its single visible GPU to index 0 for clients; setting client `=2` makes MPS reject the client ("Client requested 2 which is not a valid GPU ID in the MPS visible set"), `torch.cuda.is_available()` returns False, gets cached by transformers' `is_torch_cuda_available` `@lru_cache`, and the FA2 check then raises a misleading `"FlashAttention2 is not available on CPU"`.
  - **Use numeric index, not UUID, for the daemon.** `CUDA_VISIBLE_DEVICES=GPU-<uuid>` works for torch clients but the MPS server's driver init fails to parse it (`Driver initialization failed with: no CUDA-capable device is detected`).
  - Ampere (sm_86) doesn't require `nvidia-smi -c EXCLUSIVE_PROCESS` (root-only); MPS works fine in default compute mode.

### v0_compile
- Piecewise CUDA graph: `LlamaAttention.forward` + `create_causal_mask` disabled from compile → attention runs eagerly; MLP/norms/embed/lm_head are CUDA-graphed.
- DynamicCache batch=2 for both inference requests. Prefill uncompiled; decode compiled at static `(2, 1)` input shape.
- Same compiled `PeftModel` is used for both decode (infer via `GBMM_STEP_ADAPTERS` path) and train (`set_adapter("ft")`). Dynamo specializes on the `GBMM_STEP_ADAPTERS` flag so each mode gets its own cached compile (amortized after warmup).

### v3_compile
- Same SGMV kernels as v3.py (two adapter pointers + tl.where routing).
- KV cache for infer path: `(2, cache_len, num_kv_heads, head_dim)`, grows by 1 per decode step (one token per request).
- `@torch.compiler.disable` on `attn_forward` (flash_attn + SGMV + KV cache mutation run eagerly).
- `torch.compile(coserving_forward, mode="reduce-overhead")` → CUDA graph for embed, layernorms, MLP, lm_head between the disabled attention blocks.
- Two `fused_linear` variants: `fused_linear_attn` (SGMV + `FusedBaseLinear`, eager, inside disabled attn) and `fused_linear_mlp` (plain `proj(cat)` with `B_i*L_i` flatten/unflatten, compile-friendly) for MLP + lm_head.
- Static decode fused shape: `(1, 2*1 + 300) = (1, 302)` — constant across decode steps, so the CUDA graph replays cleanly.
- Segment tensors are rebuilt each iteration via `_set_segments(...)` in the main loop (outside the compiled region) and attached to `FusedLinearSGMV` class attrs; keeps `torch.tensor([...])` out of the Dynamo trace while still letting the kernel see up-to-date metadata on each call.
- Preserves every v3 optimization: direct infer access, bf16 backward with recomputed train_mid, train-only backward, 2-Triton-launch LoRA.
