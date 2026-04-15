## Project: LLMStation Fusion Engine

Benchmarking co-serving (simultaneous inference + LoRA fine-tuning) on a single GPU.

**Model:** meta-llama/Llama-3.2-3B, bf16
**GPU:** NVIDIA L4 (22GB), 64-core CPU, 251GB RAM
**Conda env:** llmstation (Python 3.14.3, PyTorch 2.11.0+cu130)
**Attention:** All files use Flash Attention 2 (flash-attn 2.8.3)

## Files

| File | Role |
|------|------|
| `hf_inference.py` | Pure inference benchmark (128 decode steps, `attn_implementation="flash_attention_2"`) |
| `hf_peft.py` | Pure LoRA training benchmark (128 steps, Alpaca, seq_len=300, `attn_implementation="flash_attention_2"`) |
| `hf_coserving_v0.py` | Naive co-serving: alternates decode (infer adapter) and train (ft adapter) separately (`attn_implementation="flash_attention_2"`) |
| `hf_coserving_v0s.py` | Multiprocessing co-serving: spawn + CUDA IPC shared GPU memory, decode \|\| train_fwd (parallel) → bwd+opt (sequential) |
| `hf_coserving_v1.py` | Fused co-serving: manual forward with matrix-level fusion (one HBM weight read for both paths), `attn_implementation="flash_attention_2"` + `flash_attn_func` directly |
| `hf_coserving_v2.py` | Fused co-serving + custom backward: same fused forward as v1, but `FusedBaseLinear` autograd Function skips inference tokens in backward (train-only gradient) |
| `hf_coserving_v3.py` | Fused co-serving + SGMV Triton kernels: replaces 4 cuBLAS LoRA calls per projection with 2 Triton kernel launches (shrink + expand) for both adapters, custom backward preserved |
| `hf_coserving_v0_compile.py` | v0 + torch.compile piecewise CUDA graph + KV cache for decode (DynamicCache, 1 token/step) |
| `hf_coserving_v3_compile.py` | v3 + torch.compile piecewise CUDA graph + KV cache for infer path, static fused shape (1, 301, H) |
| `compare_weights.py` | Compares step-30 LoRA checkpoints across peft/v0/v0s/v1/v2/v3 |
| `requirements.txt` | torch, transformers, accelerate, peft, datasets, ninja, flash-attn, einops, triton |

## LoRA config (all training files)
- r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.0
- v0/v0s/v1/v2/v3 use dual adapters: "ft" (train, grad) + "infer" (decode, no_grad)

## Key results (128 steps, seq_len=300)

**Speed (avg per step, ms):**

| Version | decode | train_fwd | fused_fwd | bwd+opt | total | vs baseline |
|---------|--------|-----------|-----------|---------|-------|-------------|
| inference | 38.59 | — | — | — | 38.59 | (ref) |
| peft | — | — | 64.76 | 74.56 | 139.32 | (ref) |
| v0 | 41.79 | 59.50 | — | 73.62 | 174.91 | baseline |
| v0s | wall 107.99 | (parallel) | — | 70.89 | 178.88 | +2.3% |
| v1 | — | — | 73.70 | 80.87 | 154.56 | −11.6% |
| v2 | — | — | 73.41 | 72.48 | 145.89 | −16.6% |
| v3 | — | — | 70.39 | 71.93 | 142.32 | −18.6% |
| v0_compile | 33.12 | 52.83 | — | 65.11 | 151.06 | baseline_c |
| v3_compile | — | — | 55.75 | 66.39 | 122.13 | −19.1% |

- Non-compile: after 3 warmup steps. Compile: after 5 warmup, KV cache for decode.
- Baselines: v0 (naive co-serving) and v0_compile (naive co-serving + compile). inference/peft are reference only.
- peft_compile = v0_compile train_fwd + bwd = 117.94 ms (same compiled HF forward with ft adapter)
- v0s parallel overlap saves 36.7% vs sequential, but HBM contention inflates each op ~2.5×, net slower than v0
- v2 backward is 10.4% faster than v1 (skips inference token gradients)
- v3 forward is 4.1% faster than v2 (SGMV Triton fuses 224 cuBLAS → 112 Triton kernel launches)
- v3 backward matches v2 (bf16 LoRA gradients, recomputed mid, no large fp32 casts)
- v0 → v0_compile: decode −20.7% (KV cache), train_fwd −11.2% (CUDA graph)
- v3 → v3_compile: fused_fwd −20.8% (KV cache shrinks infer from growing seq to 1 token)

**Weight equivalence (step 30):**
- peft vs v0: max_diff=5.47e-04, ALL PASS
- peft vs v0s: max_diff=5.91e-04, ALL PASS
- peft vs v1: max_diff=5.19e-04, ALL PASS
- peft vs v2: max_diff=5.51e-04, ALL PASS
- peft vs v3: max_diff=1.29e-03, ALL PASS
- v2 vs v3: max_diff=1.22e-03, ALL PASS
- All 112 LoRA params within atol=5e-3

**Loss (step 30):** peft=0.9056, v0=0.9029, v0s=0.9075, v1=0.9062, v2=0.9062, v3=0.9062

## Why each version is faster than the previous

### v0s: multiprocessing with shared GPU (decode || train_fwd parallel, +3.97 ms vs v0)
v0s uses Python multiprocessing (spawn) with CUDA IPC to share the same physical GPU tensors between two processes. Both decode (infer adapter) and train_forward (ft adapter) run in parallel, then backward+optimizer runs sequentially. The 36.7% overlap saving is real, but HBM bandwidth contention between the two processes inflates each operation ~2.5× (decode: 104.38 vs 41.79 ms solo, train_fwd: 107.25 vs 59.50 ms solo). Net result: 178.88 ms, slightly slower than v0's 174.91 ms sequential. This demonstrates that co-execution of memory-bound kernels cannot beat sequential execution.

### v0 → v1: matrix-level fusion (−20.35 ms, −11.6%)
v0 runs inference and training as two completely separate forward passes, reading every base weight matrix from HBM twice per layer. v1 concatenates inference and training tokens along the sequence dimension (`torch.cat`) and performs a single base weight matmul for both paths. This halves HBM bandwidth for the dominant base-weight loads (3072×3072 per projection). The fused forward (73.70 ms) is slower than v0's decode alone (41.79 ms) but replaces both decode + train_fwd (41.79 + 59.50 = 101.29 ms), saving 27.59 ms in forward. The backward is slightly slower (80.87 vs 73.62 ms) because autograd still processes the full concatenated (L_i + L_t) dimension.

### v1 → v2: custom backward skips inference tokens (−8.67 ms, −5.6%)
v1's autograd backward computes gradients for the full (L_i + L_t) concatenated input even though inference gradients are zero (the infer path is detached). `FusedBaseLinear(torch.autograd.Function)` keeps the same fused forward but overrides backward to only compute `grad_train @ weight` on the L_t training tokens, completely skipping the L_i inference tokens. This makes backward cost match v0's pure-training backward (72.48 vs 73.62 ms) while keeping v1's fused forward. The saving grows with the L_i/L_t ratio.

### v2 → v3: SGMV Triton kernels + bf16 backward (−3.57 ms, −2.4%)
v2 applies each adapter's LoRA A and B matrices via separate `nn.Linear` calls — 4 cuBLAS kernel launches per LoRA projection (A_infer, B_infer, A_ft, B_ft), plus Python-side dtype casts and scaling ops. With 56 LoRA projections (28 layers × 2 modules), that is 224 small cuBLAS launches per forward. v3 replaces these with 2 custom Triton kernel launches per projection (shrink: x @ A^T × scaling, expand: mid @ B^T + base_out), each processing both adapters in parallel via grid dim = adapter_id with stacked weight tensors (2, R, H) / (2, H_out, R). Pre-allocated bf16 stacked buffers (keyed by weight shape) are updated in-place via `.copy_()`, avoiding per-call `torch.stack` allocation. This reduces total kernel launches from 224 to 112 per forward, cutting forward time by 4.1% (70.39 vs 73.41 ms). The backward uses bf16 LoRA gradient computation with recomputed train_mid (avoiding 168 large fp32 casts per backward and 56 extra saved tensors in the autograd graph), matching v2's backward cost (71.93 vs 72.48 ms).

## Architecture notes

### v0s multiprocessing
- Spawn start method + CUDA IPC handles for shared GPU memory (480 tensors: params + buffers)
- Decode worker: builds model architecture on CPU from HF cache, replaces all params/buffers with shared GPU tensors via `param.data = shared_state[name]`
- Schedule: dispatch decode to worker (non-blocking) + run train_forward in main process simultaneously → wait for both → backward+optimizer in main process

### v1 fusion
- `fused_linear()`: cats infer+train along seq dim → single base weight matmul → split → detach infer path. LoRA adapters applied separately per path.
- `attn_forward()`: fused QKV/O projections, separate RoPE + flash_attn_func (handles GQA natively, no repeat_interleave)
- `mlp_forward()`: fused gate/up/down projections, separate SiLU activation

### v2 custom backward
- `FusedBaseLinear(torch.autograd.Function)`: forward identical to v1 (fused matmul), backward only computes `dL/d(train_x) = grad_train @ weight` on L_t tokens, skipping L_i inference tokens entirely.

### v3 SGMV Triton kernels
- `_sgmv_shrink_kernel`: fused x @ A^T * scaling for both adapters in one Triton launch. Grid dim 1 = adapter_id. Stacked bf16 weights: (2, R, H).
- `_sgmv_expand_kernel`: fused mid @ B^T for both adapters in one Triton launch, with ADD_INPUTS to accumulate onto base output. Stacked bf16 weights: (2, H_out, R).
- `FusedLinearSGMV(torch.autograd.Function)`: forward = 1 cuBLAS (fused base) + 2 Triton (shrink + expand). Backward = bf16 LoRA gradients with recomputed train_mid (no large fp32 casts, 4 saved tensors instead of 5), matching v2 backward cost.
- Pre-allocated bf16 stacked weight buffers (`_buf_cache`), updated via `.copy_()`. Buffers on class attribute (not through autograd) to avoid graph retention.

### v0_compile: torch.compile + KV cache
- Piecewise CUDA graph: `LlamaAttention.forward` + `create_causal_mask` disabled from compile → attention runs eagerly, MLP/norms get CUDA-graphed.
- DynamicCache for decode: prefill (step 0) fills cache with full prompt uncompiled, steps 1+ decode 1 token/step via compiled HF forward with `past_key_values`.
- `torch.compile(model, mode="reduce-overhead")` on the PEFT model — same compiled object for both decode (infer adapter) and train (ft adapter) via `set_adapter()`.
- Decode static shape: `(1, 1)` input + KV cache. Train static shape: `(1, 300)`.

### v3_compile: v3 SGMV + torch.compile + KV cache
- Same scalar-arg SGMV Triton kernels as v3.py (2 adapters: infer + ft).
- KV cache for infer path: `(1, cache_len, num_kv_heads, head_dim)`, grows by 1 token/step.
- `@torch.compiler.disable` on `attn_forward` (contains flash_attn, SGMV Triton, KV cache mutation).
- `torch.compile(coserving_forward, mode="reduce-overhead")` → CUDA graph for MLP/norms between disabled attention blocks.
- Two `fused_linear` variants: `fused_linear_attn` (SGMV + FusedBaseLinear, eager) for attn projections, `fused_linear_mlp` (plain `proj(cat)`, compiled) for MLP + lm_head.
- Static decode fused shape: `(1, 1+300)` = `(1, 301)` for all decode steps. Prefill shape: `(1, prompt_len+300)`.
- v3_compile preserves all v3 optimizations: FusedLinearSGMV (bf16 backward, recomputed train_mid), FusedBaseLinear (train-only backward), SGMV kernel launch reduction (224→112 cuBLAS → Triton).
