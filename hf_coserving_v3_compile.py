"""Fused co-serving v3 with torch.compile piecewise CUDA graph.

Same as hf_coserving_v3.py (SGMV Triton kernels, 2 adapters) but uses:
  - KV cache for infer path: prefill full prompt, decode 1 token/step
  - @torch.compiler.disable on attn_forward (flash_attn, SGMV Triton, KV cache)
  - torch.compile(coserving_forward, mode="reduce-overhead") -> CUDA graph for MLP/norms
  - Static fused matmul shape: (1, 1+300, H) = (1, 301, H) for all decode steps
  - Prefill (step 0) runs uncompiled, decode (steps 1+) runs compiled

v3 optimizations preserved:
  - FusedLinearSGMV: SGMV Triton (shrink + expand) for both adapters in attn_forward
  - FusedBaseLinear: train-only backward for non-LoRA projections (k_proj, o_proj)
  - bf16 LoRA backward with recomputed train_mid
  - MLP uses compile-friendly fused_linear_mlp (no custom autograd)
"""
import time
import torch
torch._dynamo.config.cache_size_limit = 64

import torch.nn.functional as F
import triton
import triton.language as tl
from flash_attn import flash_attn_func
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

torch.manual_seed(42)
torch.cuda.manual_seed(42)

model_name = "meta-llama/Llama-3.2-3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map="auto",
                                             attn_implementation="flash_attention_2")

# Same 2 adapters as v3: ft (train) + infer (decode)
lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.0)
model = get_peft_model(model, lora_config, adapter_name="ft")
model.add_adapter("infer", lora_config)
model.set_adapter("ft")
model.print_trainable_parameters()

prompt = "Explain what machine learning is in one sentence."
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

max_new_tokens = 128
warmup_steps = 5

# Pre-tokenize training samples from Alpaca, seq_len=300
dataset = load_dataset("tatsu-lab/alpaca", split="train")
train_all_ids = []
for i in range(max_new_tokens):
    s = dataset[i]
    text = f"### Instruction:\n{s['instruction']}\n\n### Input:\n{s['input']}\n\n### Response:\n{s['output']}"
    ids = tokenizer(text, return_tensors="pt", max_length=300, padding="max_length", truncation=True).input_ids
    train_all_ids.append(ids)
train_all_ids = torch.cat(train_all_ids, dim=0).to(model.device)

optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

# Navigate PEFT wrapper
base = model.base_model.model.model
lm_head = model.base_model.model.lm_head
cfg = model.config
num_heads = cfg.num_attention_heads
num_kv_heads = cfg.num_key_value_heads
head_dim = cfg.hidden_size // num_heads

device = model.device
prompt_len = input_ids.shape[1]
num_layers = len(base.layers)

# KV cache for infer path: (1, cache_len, num_kv_heads, head_dim)
_kv_cache_k = [None] * num_layers
_kv_cache_v = [None] * num_layers


# ============================================================================
# SGMV Triton Kernels — scalar args for 2 adapters (matching v3.py)
# ============================================================================

@triton.jit
def _sgmv_shrink_kernel(
    x_ptr, a_ptr, out_ptr,
    seg_start_0, seg_len_0, scaling_0,
    seg_start_1, seg_len_1, scaling_1,
    stride_x_s, stride_x_h,
    stride_a_a, stride_a_r, stride_a_h,
    stride_o_s, stride_o_r,
    HIDDEN: tl.constexpr,
    RANK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    adapter_id = tl.program_id(1)

    if adapter_id == 0:
        seg_start = seg_start_0
        seg_len = seg_len_0
        scaling = scaling_0
    else:
        seg_start = seg_start_1
        seg_len = seg_len_1
        scaling = scaling_1

    if pid_m * BLOCK_M >= seg_len:
        return

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < seg_len
    global_m = seg_start + offs_m

    offs_n = tl.arange(0, BLOCK_N)
    mask_n = offs_n < RANK

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, HIDDEN, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        mask_k = offs_k < HIDDEN
        x = tl.load(
            x_ptr + global_m[:, None] * stride_x_s + offs_k[None, :] * stride_x_h,
            mask=mask_m[:, None] & mask_k[None, :], other=0.0,
        )
        a = tl.load(
            a_ptr + adapter_id * stride_a_a
            + offs_k[:, None] * stride_a_h + offs_n[None, :] * stride_a_r,
            mask=mask_k[:, None] & mask_n[None, :], other=0.0,
        )
        acc += tl.dot(x, a)

    acc *= scaling

    tl.store(
        out_ptr + global_m[:, None] * stride_o_s + offs_n[None, :] * stride_o_r,
        acc.to(out_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_n[None, :],
    )


@triton.jit
def _sgmv_expand_kernel(
    mid_ptr, b_ptr, out_ptr,
    seg_start_0, seg_len_0,
    seg_start_1, seg_len_1,
    stride_m_s, stride_m_r,
    stride_b_a, stride_b_h, stride_b_r,
    stride_o_s, stride_o_h,
    HIDDEN_OUT: tl.constexpr,
    RANK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ADD_INPUTS: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    adapter_id = tl.program_id(2)

    if adapter_id == 0:
        seg_start = seg_start_0
        seg_len = seg_len_0
    else:
        seg_start = seg_start_1
        seg_len = seg_len_1

    if pid_m * BLOCK_M >= seg_len:
        return

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < seg_len
    global_m = seg_start + offs_m

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < HIDDEN_OUT

    offs_k = tl.arange(0, BLOCK_K)
    mask_k = offs_k < RANK

    mid = tl.load(
        mid_ptr + global_m[:, None] * stride_m_s + offs_k[None, :] * stride_m_r,
        mask=mask_m[:, None] & mask_k[None, :], other=0.0,
    )
    b = tl.load(
        b_ptr + adapter_id * stride_b_a
        + offs_k[:, None] * stride_b_r + offs_n[None, :] * stride_b_h,
        mask=mask_k[:, None] & mask_n[None, :], other=0.0,
    )

    acc = tl.dot(mid, b)

    c_ptrs = out_ptr + global_m[:, None] * stride_o_s + offs_n[None, :] * stride_o_h
    c_mask = mask_m[:, None] & mask_n[None, :]

    if ADD_INPUTS:
        existing = tl.load(c_ptrs, mask=c_mask, other=0.0)
        acc += existing.to(tl.float32)

    tl.store(c_ptrs, acc.to(out_ptr.dtype.element_ty), mask=c_mask)


# ============================================================================
# Python wrappers + pre-allocated stacked weight buffers (matching v3.py)
# ============================================================================

_BLOCK_M = 32
_BLOCK_N_SHRINK = 16
_BLOCK_K_SHRINK = 128
_BLOCK_N_EXPAND = 128
_BLOCK_K_EXPAND = 16

_buf_cache = {}


def _get_bufs(a_shape, b_shape, device):
    key = (a_shape, b_shape)
    if key not in _buf_cache:
        _buf_cache[key] = (
            torch.empty(2, *a_shape, dtype=torch.bfloat16, device=device),
            torch.empty(2, *b_shape, dtype=torch.bfloat16, device=device),
        )
    return _buf_cache[key]


def _update_bufs(a_buf, b_buf, a_infer, a_ft, b_infer, b_ft):
    a_buf[0].copy_(a_infer)
    a_buf[1].copy_(a_ft)
    b_buf[0].copy_(b_infer)
    b_buf[1].copy_(b_ft)


def sgmv_shrink(x, a_buf, L_i, L_t, scaling):
    _, L_total, H = x.shape
    R = a_buf.shape[1]
    out = torch.empty(1, L_total, R, dtype=x.dtype, device=x.device)
    max_seg = max(L_i, L_t)

    _sgmv_shrink_kernel[(triton.cdiv(max_seg, _BLOCK_M), 2)](
        x, a_buf, out,
        0, L_i, scaling,
        L_i, L_t, scaling,
        x.stride(1), x.stride(2),
        a_buf.stride(0), a_buf.stride(1), a_buf.stride(2),
        out.stride(1), out.stride(2),
        HIDDEN=H, RANK=R,
        BLOCK_M=_BLOCK_M, BLOCK_N=_BLOCK_N_SHRINK, BLOCK_K=_BLOCK_K_SHRINK,
    )
    return out


def sgmv_expand(mid, b_buf, L_i, L_t, out):
    _, L_total, R = mid.shape
    H_out = b_buf.shape[1]

    _sgmv_expand_kernel[(triton.cdiv(max(L_i, L_t), _BLOCK_M), triton.cdiv(H_out, _BLOCK_N_EXPAND), 2)](
        mid, b_buf, out,
        0, L_i,
        L_i, L_t,
        mid.stride(1), mid.stride(2),
        b_buf.stride(0), b_buf.stride(1), b_buf.stride(2),
        out.stride(1), out.stride(2),
        HIDDEN_OUT=H_out, RANK=R,
        BLOCK_M=_BLOCK_M, BLOCK_N=_BLOCK_N_EXPAND, BLOCK_K=_BLOCK_K_EXPAND,
        ADD_INPUTS=True,
    )


# ============================================================================
# Custom Autograd Functions
# ============================================================================

class FusedBaseLinear(torch.autograd.Function):
    """Train-only backward for non-LoRA attn projections (k_proj, o_proj)."""
    @staticmethod
    def forward(ctx, train_x, weight, bias, infer_x):
        L_i = infer_x.shape[1]
        combined = torch.cat([infer_x, train_x], dim=1)
        out = F.linear(combined, weight, bias)
        ctx.save_for_backward(weight)
        return out[:, :L_i], out[:, L_i:]

    @staticmethod
    def backward(ctx, grad_infer, grad_train):
        weight, = ctx.saved_tensors
        return grad_train @ weight, None, None, None


class FusedLinearSGMV(torch.autograd.Function):
    """Fused base matmul + SGMV LoRA for both adapters, train-only backward.
    Forward: 1 cuBLAS (base) + 2 Triton (shrink + expand).
    Backward: bf16 LoRA gradients with recomputed train_mid.
    """
    _a_buf = None
    _b_buf = None

    @staticmethod
    def forward(ctx, train_x, infer_x, base_weight, base_bias,
                A_ft_w, B_ft_w, scaling):
        L_i = infer_x.shape[1]
        L_t = train_x.shape[1]

        combined = torch.cat([infer_x, train_x], dim=1)
        base_out = F.linear(combined, base_weight, base_bias)

        a_buf = FusedLinearSGMV._a_buf
        b_buf = FusedLinearSGMV._b_buf

        mid = sgmv_shrink(combined, a_buf, L_i, L_t, scaling)
        sgmv_expand(mid, b_buf, L_i, L_t, base_out)

        ctx.save_for_backward(train_x, base_weight, A_ft_w, B_ft_w)
        ctx.scaling = scaling

        return base_out[:, :L_i], base_out[:, L_i:]

    @staticmethod
    def backward(ctx, grad_infer, grad_train):
        train_x, base_weight, A_ft_w, B_ft_w = ctx.saved_tensors
        scaling = ctx.scaling

        grad_train_x = grad_train @ base_weight

        A_bf16 = A_ft_w.bfloat16()
        B_bf16 = B_ft_w.bfloat16()

        train_mid = F.linear(train_x, A_bf16) * scaling
        grad_mid = grad_train @ B_bf16
        grad_B = (grad_train.squeeze(0).T @ train_mid.squeeze(0)).float()
        grad_z = grad_mid * scaling
        grad_train_x = grad_train_x + grad_z @ A_bf16
        grad_A = (grad_z.squeeze(0).T @ train_x.squeeze(0)).float()

        return grad_train_x, None, None, None, grad_A, grad_B, None


# --------------------------------------------------------------------------
# Two fused_linear variants:
#   fused_linear_attn: SGMV + FusedBaseLinear (eager, in disabled attn)
#   fused_linear_mlp:  compile-friendly proj(cat) (compiled MLP + lm_head)
# --------------------------------------------------------------------------

def fused_linear_attn(proj, infer_x, train_x):
    """SGMV Triton for LoRA (q_proj, v_proj), FusedBaseLinear for non-LoRA (k_proj, o_proj)."""
    if hasattr(proj, 'base_layer'):
        base_layer = proj.base_layer
        L_i = infer_x.shape[1]
        L_t = train_x.shape[1]

        a_buf, b_buf = _get_bufs(
            proj.lora_A["infer"].weight.shape,
            proj.lora_B["infer"].weight.shape,
            infer_x.device,
        )
        _update_bufs(a_buf, b_buf,
                     proj.lora_A["infer"].weight, proj.lora_A["ft"].weight,
                     proj.lora_B["infer"].weight, proj.lora_B["ft"].weight)

        FusedLinearSGMV._a_buf = a_buf
        FusedLinearSGMV._b_buf = b_buf

        infer_out, train_out = FusedLinearSGMV.apply(
            train_x, infer_x,
            base_layer.weight, base_layer.bias,
            proj.lora_A["ft"].weight, proj.lora_B["ft"].weight,
            proj.scaling["ft"],
        )
        return infer_out.detach(), train_out
    else:
        infer_out, train_out = FusedBaseLinear.apply(
            train_x, proj.weight, proj.bias, infer_x
        )
        return infer_out.detach(), train_out


def fused_linear_mlp(proj, infer_x, train_x):
    """Compile-friendly fused linear for MLP + lm_head."""
    L_i = infer_x.shape[1]
    out = proj(torch.cat([infer_x, train_x], dim=1))
    return out[:, :L_i].detach(), out[:, L_i:]


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, cos, sin):
    cos = cos.unsqueeze(2)
    sin = sin.unsqueeze(2)
    return (q * cos + rotate_half(q) * sin), (k * cos + rotate_half(k) * sin)


@torch.compiler.disable
def attn_forward(attn, infer_h, train_h, infer_rope, train_rope, layer_idx):
    """Flash Attention with SGMV LoRA + KV cache for infer, no KV cache for train."""
    L_i, L_t = infer_h.shape[1], train_h.shape[1]

    infer_q, train_q = fused_linear_attn(attn.q_proj, infer_h, train_h)
    infer_k, train_k = fused_linear_attn(attn.k_proj, infer_h, train_h)
    infer_v, train_v = fused_linear_attn(attn.v_proj, infer_h, train_h)

    # Infer path: (1, L_i, ...) with KV cache
    with torch.no_grad():
        infer_q = infer_q.view(1, L_i, num_heads, head_dim)
        infer_k = infer_k.view(1, L_i, num_kv_heads, head_dim)
        infer_v = infer_v.view(1, L_i, num_kv_heads, head_dim)
        cos_i, sin_i = infer_rope
        infer_q, infer_k = apply_rope(infer_q, infer_k, cos_i, sin_i)

        if _kv_cache_k[layer_idx] is not None:
            infer_k = torch.cat([_kv_cache_k[layer_idx], infer_k], dim=1)
            infer_v = torch.cat([_kv_cache_v[layer_idx], infer_v], dim=1)
        _kv_cache_k[layer_idx] = infer_k
        _kv_cache_v[layer_idx] = infer_v

        infer_attn = flash_attn_func(infer_q, infer_k, infer_v, causal=True)
        infer_attn = infer_attn.reshape(1, L_i, -1)

    # Train path: (1, L_t, ...) no KV cache
    train_q = train_q.view(1, L_t, num_heads, head_dim)
    train_k = train_k.view(1, L_t, num_kv_heads, head_dim)
    train_v = train_v.view(1, L_t, num_kv_heads, head_dim)
    cos_t, sin_t = train_rope
    train_q, train_k = apply_rope(train_q, train_k, cos_t, sin_t)
    train_attn = flash_attn_func(train_q, train_k, train_v, causal=True)
    train_attn = train_attn.reshape(1, L_t, -1)

    infer_out, train_out = fused_linear_attn(attn.o_proj, infer_attn, train_attn)
    return infer_out, train_out


def mlp_forward(layer, infer_h, train_h):
    mlp = layer.mlp
    infer_n = layer.post_attention_layernorm(infer_h)
    train_n = layer.post_attention_layernorm(train_h)

    infer_gate, train_gate = fused_linear_mlp(mlp.gate_proj, infer_n, train_n)
    infer_up,   train_up   = fused_linear_mlp(mlp.up_proj,   infer_n, train_n)

    with torch.no_grad():
        infer_act = F.silu(infer_gate) * infer_up
    train_act = F.silu(train_gate) * train_up

    infer_down, train_down = fused_linear_mlp(mlp.down_proj, infer_act, train_act)
    return infer_h + infer_down, train_h + train_down


def coserving_forward(infer_ids, train_ids, infer_pos):
    infer_h = base.embed_tokens(infer_ids).detach()
    train_h = base.embed_tokens(train_ids)

    L_t = train_ids.shape[1]
    train_pos = torch.arange(L_t, device=device).unsqueeze(0)

    rotary = base.rotary_emb
    infer_rope = rotary(infer_h, infer_pos)
    train_rope = rotary(train_h, train_pos)

    for i, layer in enumerate(base.layers):
        infer_n = layer.input_layernorm(infer_h)
        train_n = layer.input_layernorm(train_h)
        infer_a, train_a = attn_forward(layer.self_attn, infer_n, train_n,
                                         infer_rope, train_rope, layer_idx=i)
        infer_h = infer_h + infer_a
        train_h = train_h + train_a
        infer_h, train_h = mlp_forward(layer, infer_h, train_h)

    infer_h = base.norm(infer_h).detach()
    train_h = base.norm(train_h)
    infer_logits, train_logits = fused_linear_mlp(lm_head, infer_h, train_h)
    return infer_logits, train_logits


# --------------------------------------------------------------------------
# Prefill (step 0, uncompiled)
# --------------------------------------------------------------------------
print("[prefill] Running uncompiled step 0 (prefill + train)...")
model.train()

train_ids_0 = train_all_ids[0:1]
infer_ids_prefill = input_ids  # (1, prompt_len)
infer_pos_0 = torch.arange(prompt_len, device=device).unsqueeze(0)

torch.cuda.synchronize()
t0 = time.perf_counter()
infer_logits, train_logits = coserving_forward(infer_ids_prefill, train_ids_0, infer_pos_0)
torch.cuda.synchronize()
t1 = time.perf_counter()

next_token = infer_logits[0, -1, :].argmax().reshape(1, 1)
generated_tokens = [next_token]

shift_logits = train_logits[:, :-1, :].contiguous()
shift_labels = train_ids_0[:, 1:].contiguous()
loss = F.cross_entropy(shift_logits.view(-1, shift_logits.shape[-1]), shift_labels.view(-1))

torch.cuda.synchronize()
t2 = time.perf_counter()
loss.backward()
optimizer.step()
optimizer.zero_grad()
torch.cuda.synchronize()
t3 = time.perf_counter()

print(f"step   0 | loss={loss.item():.4f} | fused_fwd={t1-t0:.4f}s | bwd+opt={t3-t2:.4f}s (prefill, uncompiled)")
print(f"[prefill] Done. KV cache shape: {_kv_cache_k[0].shape}")

# --------------------------------------------------------------------------
# Compile for decode
# --------------------------------------------------------------------------
compiled_forward = torch.compile(coserving_forward, mode="reduce-overhead")
print(f"[compile] reduce-overhead, piecewise (attn_forward disabled)")
print(f"[compile] Static decode shape: infer=(1, 1), train=(1, 300), fused=(1, 301)")

# --------------------------------------------------------------------------
# Decode loop (steps 1+)
# --------------------------------------------------------------------------
fused_fwd_times = []
bwd_opt_times = []

for step in range(1, max_new_tokens):
    model.train()
    train_ids = train_all_ids[step:step+1]

    infer_ids = next_token  # (1, 1)
    infer_pos = torch.full((1, 1), prompt_len + step - 1, device=device, dtype=torch.long)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    infer_logits, train_logits = compiled_forward(infer_ids, train_ids, infer_pos)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    next_token = infer_logits[0, -1, :].argmax().reshape(1, 1)
    generated_tokens.append(next_token)

    if next_token.item() == tokenizer.eos_token_id:
        break

    shift_logits = train_logits[:, :-1, :].contiguous()
    shift_labels = train_ids[:, 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]),
        shift_labels.view(-1),
    )

    torch.cuda.synchronize()
    t2 = time.perf_counter()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    torch.cuda.synchronize()
    t3 = time.perf_counter()

    if step >= warmup_steps:
        fused_fwd_times.append(t1 - t0)
        bwd_opt_times.append(t3 - t2)

    if step == 30:
        ckpt = {k: v.data.clone().cpu() for k, v in model.named_parameters() if v.requires_grad}
        torch.save(ckpt, "weights_v3_compile_fa_step30.pt")
        print(f"[CHECKPOINT] step 30 weight sum: {sum(v.sum().item() for v in ckpt.values()):.10f}")

    if step % 10 == 0:
        print(f"step {step:3d} | loss={loss.item():.4f} | fused_fwd={t1-t0:.4f}s | bwd+opt={t3-t2:.4f}s")

# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------
all_tokens = torch.cat([input_ids] + generated_tokens, dim=-1)
response = tokenizer.decode(all_tokens[0][prompt_len:], skip_special_tokens=True)
print(f"\n--- Generated response ---\n{response}")

if fused_fwd_times:
    avg_fused = sum(fused_fwd_times) / len(fused_fwd_times)
    avg_bwd = sum(bwd_opt_times) / len(bwd_opt_times)
    print(f"\n--- Timing (after {warmup_steps} warmup, {len(fused_fwd_times)} measured steps) ---")
    print(f"  Decode fused shape: (1, 301) = (1, 1 infer + 300 train)")
    print(f"  KV cache final: {_kv_cache_k[0].shape[1]} tokens x {num_layers} layers")
    print(f"  avg fused_fwd (decode+peft): {avg_fused*1000:.2f} ms")
    print(f"  avg backward+optimizer:      {avg_bwd*1000:.2f} ms")
    print(f"  avg total:                   {(avg_fused+avg_bwd)*1000:.2f} ms")
