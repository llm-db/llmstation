## Project: LLMStation Fusion Engine

Benchmarking co-serving (simultaneous multi-request inference + LoRA fine-tuning) on a single GPU.

- **Model:** meta-llama/Llama-3.2-3B, bf16
- **GPU:** NVIDIA L4 (22GB), 64-core CPU, 251GB RAM
- **Conda env:** llmstation (Python 3.14.3, PyTorch 2.11.0+cu130)
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
| inference (1 req, KV cache) | 31.23 | — | — | 31.23 | (ref) |
| peft (train only) | — | 63.14 | 72.96 | 136.10 | (ref) |
| v0 (gather-BMM naive) | 34.74 | 58.35 | 72.98 | 166.07 | baseline |
| v0s (gather-BMM, decode‖train) | parallel wall 97.74 | — | 72.15 | 169.89 | +2.3% |
| v1 (gather-BMM + fusion) | 62.88 | — | 72.47 | 135.36 | −18.5% |
| v2 (+ train-only backward) | 63.56 | — | 70.94 | 134.50 | −19.0% |
| v3 (SGMV, direct infer access) | 59.82 | — | 69.48 | **129.31** | −22.1% |
| v0_compile (gather-BMM + compile) | 32.31 | 54.16 | 68.75 | 155.22 | baseline_c |
| v3_compile (SGMV + compile) | 55.18 | — | 65.27 | **120.46** | −22.4% |

Notes:
- Baselines: **v0** (non-compile) and **v0_compile** (compile). peft is a training-only reference and is not treated as a baseline.
- In v0/v0s/v0_compile the decode is a standalone batched `(2, 1)` forward and train_fwd is a separate call; in v1/v2/v3/v3_compile both are concatenated into one fused matmul (`fused_fwd`).
- v0s loses vs v0 sequential: HBM contention inflates both decode and train_fwd ~1.5–2× when run in parallel, eating more time than the overlap saves.
- v3 is the fastest non-compile variant (−22.1% vs v0); v3_compile is the fastest overall (−22.4% vs v0_compile).

**Weight equivalence (step-30 ft-adapter params vs pure peft, atol=5e-3):**

| Comparison | max_diff | Status |
|---|---|---|
| peft vs v0 | 6.29e-04 | ALL PASS |
| peft vs v0s | 4.91e-04 | ALL PASS |
| peft vs v0_compile | 2.13e-03 | ALL PASS |
| peft vs v1 | 8.72e-04 | ALL PASS |
| peft vs v2 | 9.39e-04 | ALL PASS |
| peft vs v3 | 1.43e-03 | ALL PASS |
| peft vs v3_compile | 1.95e-03 | ALL PASS |
| v2 vs v3 | 1.35e-03 | ALL PASS |
| v3 vs v3_compile | 2.06e-03 | ALL PASS |

All 112 trainable LoRA params within atol=5e-3 across every variant.

## Why each version is faster than the previous

### v0s: multiprocessing with shared GPU (still slower than v0)
Spawn + CUDA IPC puts two processes on the same physical GPU memory and runs the batched 2-request decode in parallel with train_fwd. The overlap saving is real (~29% vs sequential), but HBM contention inflates each op substantially (decode 34.7 → 70.8 ms, train_fwd 58.4 → 97.0 ms), so net wall time ends up slightly above v0.

### v0 → v1: matrix-level fusion (−30.7 ms, −18.5% vs v0)
v0 runs decode and train as fully separate forward passes — every base weight matrix is read from HBM twice per layer. v1 flattens the inference batch `(2, L_i, H) → (1, 2*L_i, H)` and concatenates with the training tokens `(1, L_t, H)` along the sequence dim for a single base-weight matmul covering both. Halves HBM bandwidth on the dominant base-weight loads. Infer LoRA uses gather-BMM; train LoRA keeps the ft adapter via PEFT.

### v1 → v2: train-only backward (−0.86 ms, −0.6% vs v1)
v1's autograd backward still processes the full `(2*L_i + L_t)` dimension even though infer contributes zero gradient. `FusedBaseLinear(torch.autograd.Function)` keeps v1's fused forward but overrides backward to compute only `grad_train @ W` on the `L_t` training tokens, skipping the `2*L_i` infer tokens entirely. Savings are small here because L_t=300 dominates but would grow with the infer/train ratio.

### v2 → v3: SGMV with direct adapter access (−5.2 ms, −3.9% vs v2)
Two complementary properties of SGMV over gather-BMM:

1. **No per-request weight duplication.** Gather-BMM needs to stack the infer adapter weight as many times as there are inference requests (`(B_i, R, H)` for the A matrix). SGMV stores each unique adapter once, and N segments route to K ≤ N unique slots. With 2 infer requests sharing one "infer" adapter, K=2 total unique slots (infer + ft) — not 3.
2. **No extra materialized copy of the infer adapter at all.** The Triton kernel receives `proj.lora_A["infer"].weight.data_ptr()` directly. The only remaining copy is a tiny one-shape bf16 buffer for the ft adapter, needed because PEFT stores ft in fp32 and the kernel reads bf16 — this is a dtype-conversion buffer, not a duplicate for batching.

Each shrink / expand kernel takes two adapter pointers (`a_infer_ptr`, `a_ft_bf16_ptr`) and uses `tl.where(adapter_id == 0, a_infer_tile, a_ft_tile)` at block granularity to route. Still 2 Triton launches per LoRA projection regardless of request count. Forward drops ~3.7 ms; backward drops ~1.5 ms (bf16 LoRA grads with recomputed `train_mid`, matching v2 cost).

### v0 → v0_compile: compile applied to the non-compile baseline (−10.85 ms, −6.5% vs v0)
Same batched-decode + separate-train structure as v0, but with `torch.compile(mode="reduce-overhead")` + DynamicCache. Attention and mask creation run eagerly, everything else gets CUDA-graphed. Sets the compile baseline at 155.22 ms.

### v3 → v3_compile: compile applied to the best non-compile variant (−8.85 ms, −6.8% vs v3; −22.4% vs v0_compile)
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

### v0s multiprocessing
- Spawn + CUDA IPC to share ~480 tensors (params + buffers) between processes — both reference the same physical GPU memory.
- Decode worker: builds the model on CPU, then rebinds every `param.data` / `buf.data` to the shared GPU tensor; also monkey-patches PEFT's LoraLinear for gather-BMM with the worker's `step_adapters_ref`.
- Per step: main dispatches `(op, batched_tokens, [kv_pos,] step_adapters)` to worker (non-blocking), runs train_fwd locally, waits for worker's batched next tokens, then runs bwd+opt.

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
