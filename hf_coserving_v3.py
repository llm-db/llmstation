import time
import torch
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

# Two LoRA adapters: infer (shared by both inference requests) + ft (training)
lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.0)
model = get_peft_model(model, lora_config, adapter_name="ft")
model.add_adapter("infer", lora_config)
model.set_adapter("ft")  # only ft adapter requires grad
model.print_trainable_parameters()

# Two inference requests share the prompt and the infer adapter; SGMV segments
# them natively — multiple segments can index the SAME adapter slot, so "infer"
# is never duplicated (contrast with gather-BMM stacking N slot copies).
prompt = "Explain what machine learning is in one sentence."
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device).repeat(2, 1)  # (2, L)

max_new_tokens = 128
warmup_steps = 3

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
n_rep = num_heads // num_kv_heads

prompt_len = input_ids.shape[1]
num_layers = len(base.layers)
_kv_cache_k = [None] * num_layers  # each (B_i, kv_len, num_kv_heads, head_dim)
_kv_cache_v = [None] * num_layers


# --------------------------------------------------------------------------
# v3 optimization: SGMV Triton kernels — direct adapter access, no stacking.
#
# The inference adapter (bf16, frozen) is read DIRECTLY from PEFT's Parameter
# storage; no pool/stack copy is materialized. The training adapter (fp32,
# trainable) still needs a tiny bf16 buffer per shape because the kernel
# requires bf16 inputs and ft is stored in fp32 for optimizer precision —
# this is a dtype-conversion buffer, not a duplicate for batching.
#
# Each kernel takes two adapter pointers (a_infer_ptr, a_ft_bf16_ptr) and
# per-segment adapter_id ∈ {0, 1}. All segments that reference the "infer"
# adapter index the same physical memory — no per-request duplication, which
# is exactly the SGMV advantage over gather-BMM.
# --------------------------------------------------------------------------


# ============================================================================
# SGMV Triton Kernels — two adapter pointers, tl.where selection
# ============================================================================

@triton.jit
def _sgmv_shrink_kernel(
    x_ptr,
    a_infer_ptr, a_ft_ptr,
    out_ptr,
    seg_starts_ptr, seg_lens_ptr, seg_adapters_ptr, seg_scalings_ptr,
    stride_x_s, stride_x_h,
    stride_a_r, stride_a_h,
    stride_o_s, stride_o_r,
    HIDDEN: tl.constexpr,
    RANK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """SGMV shrink. Per segment, select between a_infer and a_ft by adapter_id.
    Grid: (cdiv(max_seg, BLOCK_M), N_SEGMENTS). Both adapters share strides.
    """
    pid_m = tl.program_id(0)
    seg_id = tl.program_id(1)

    seg_start = tl.load(seg_starts_ptr + seg_id)
    seg_len = tl.load(seg_lens_ptr + seg_id)
    adapter_id = tl.load(seg_adapters_ptr + seg_id)
    scaling = tl.load(seg_scalings_ptr + seg_id)

    if pid_m * BLOCK_M >= seg_len:
        return

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < seg_len
    global_m = seg_start + offs_m

    offs_n = tl.arange(0, BLOCK_N)
    mask_n = offs_n < RANK

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    is_infer = adapter_id == 0

    for k in range(0, HIDDEN, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        mask_k = offs_k < HIDDEN

        x = tl.load(
            x_ptr + global_m[:, None] * stride_x_s + offs_k[None, :] * stride_x_h,
            mask=mask_m[:, None] & mask_k[None, :], other=0.0,
        )
        a_offs = offs_k[:, None] * stride_a_h + offs_n[None, :] * stride_a_r
        a_mask = mask_k[:, None] & mask_n[None, :]
        a_infer = tl.load(a_infer_ptr + a_offs, mask=a_mask, other=0.0)
        a_ft = tl.load(a_ft_ptr + a_offs, mask=a_mask, other=0.0)
        a = tl.where(is_infer, a_infer, a_ft)
        acc += tl.dot(x, a)

    acc *= scaling

    tl.store(
        out_ptr + global_m[:, None] * stride_o_s + offs_n[None, :] * stride_o_r,
        acc.to(out_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_n[None, :],
    )


@triton.jit
def _sgmv_expand_kernel(
    mid_ptr,
    b_infer_ptr, b_ft_ptr,
    out_ptr,
    seg_starts_ptr, seg_lens_ptr, seg_adapters_ptr,
    stride_m_s, stride_m_r,
    stride_b_h, stride_b_r,
    stride_o_s, stride_o_h,
    HIDDEN_OUT: tl.constexpr,
    RANK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ADD_INPUTS: tl.constexpr,
):
    """SGMV expand. Per segment, select between b_infer and b_ft by adapter_id.
    Grid: (cdiv(max_seg, BLOCK_M), cdiv(H_out, BLOCK_N), N_SEGMENTS).
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    seg_id = tl.program_id(2)

    seg_start = tl.load(seg_starts_ptr + seg_id)
    seg_len = tl.load(seg_lens_ptr + seg_id)
    adapter_id = tl.load(seg_adapters_ptr + seg_id)

    if pid_m * BLOCK_M >= seg_len:
        return

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < seg_len
    global_m = seg_start + offs_m

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < HIDDEN_OUT

    offs_k = tl.arange(0, BLOCK_K)
    mask_k = offs_k < RANK

    is_infer = adapter_id == 0

    mid = tl.load(
        mid_ptr + global_m[:, None] * stride_m_s + offs_k[None, :] * stride_m_r,
        mask=mask_m[:, None] & mask_k[None, :], other=0.0,
    )
    b_offs = offs_k[:, None] * stride_b_r + offs_n[None, :] * stride_b_h
    b_mask = mask_k[:, None] & mask_n[None, :]
    b_infer = tl.load(b_infer_ptr + b_offs, mask=b_mask, other=0.0)
    b_ft = tl.load(b_ft_ptr + b_offs, mask=b_mask, other=0.0)
    b = tl.where(is_infer, b_infer, b_ft)

    acc = tl.dot(mid, b)

    c_ptrs = out_ptr + global_m[:, None] * stride_o_s + offs_n[None, :] * stride_o_h
    c_mask = mask_m[:, None] & mask_n[None, :]

    if ADD_INPUTS:
        existing = tl.load(c_ptrs, mask=c_mask, other=0.0)
        acc += existing.to(tl.float32)

    tl.store(c_ptrs, acc.to(out_ptr.dtype.element_ty), mask=c_mask)


# ============================================================================
# Python wrappers + minimal bf16 buffer (ft only, for fp32→bf16 cast)
# ============================================================================

_BLOCK_M = 32
_BLOCK_N_SHRINK = 16
_BLOCK_K_SHRINK = 128
_BLOCK_N_EXPAND = 128
_BLOCK_K_EXPAND = 16

# Single-slot bf16 buffers for the ft adapter (dtype conversion only).
# NO buffer for infer — it's read directly from PEFT's Parameter storage.
_ft_buf_cache = {}


def _get_ft_bufs(a_shape, b_shape, device):
    key = (a_shape, b_shape)
    if key not in _ft_buf_cache:
        _ft_buf_cache[key] = (
            torch.empty(*a_shape, dtype=torch.bfloat16, device=device),
            torch.empty(*b_shape, dtype=torch.bfloat16, device=device),
        )
    return _ft_buf_cache[key]


def sgmv_shrink(x, a_infer, a_ft_buf, seg_starts, seg_lens, seg_adapters, seg_scalings, max_seg):
    """Per-segment x @ A^T * scaling. a_infer is PEFT's Parameter (bf16, direct)."""
    _, L_total, H = x.shape
    R = a_infer.shape[0]
    out = torch.empty(1, L_total, R, dtype=x.dtype, device=x.device)
    n_segs = seg_starts.shape[0]

    _sgmv_shrink_kernel[(triton.cdiv(max_seg, _BLOCK_M), n_segs)](
        x, a_infer, a_ft_buf, out,
        seg_starts, seg_lens, seg_adapters, seg_scalings,
        x.stride(1), x.stride(2),
        a_infer.stride(0), a_infer.stride(1),  # (R, H) → stride_a_r, stride_a_h
        out.stride(1), out.stride(2),
        HIDDEN=H, RANK=R,
        BLOCK_M=_BLOCK_M, BLOCK_N=_BLOCK_N_SHRINK, BLOCK_K=_BLOCK_K_SHRINK,
    )
    return out


def sgmv_expand(mid, b_infer, b_ft_buf, seg_starts, seg_lens, seg_adapters, out, max_seg):
    """Per-segment mid @ B^T, added to out. b_infer is PEFT's Parameter (bf16, direct)."""
    _, L_total, R = mid.shape
    H_out = b_infer.shape[0]
    n_segs = seg_starts.shape[0]

    _sgmv_expand_kernel[(triton.cdiv(max_seg, _BLOCK_M), triton.cdiv(H_out, _BLOCK_N_EXPAND), n_segs)](
        mid, b_infer, b_ft_buf, out,
        seg_starts, seg_lens, seg_adapters,
        mid.stride(1), mid.stride(2),
        b_infer.stride(0), b_infer.stride(1),  # (H_out, R) → stride_b_h, stride_b_r
        out.stride(1), out.stride(2),
        HIDDEN_OUT=H_out, RANK=R,
        BLOCK_M=_BLOCK_M, BLOCK_N=_BLOCK_N_EXPAND, BLOCK_K=_BLOCK_K_EXPAND,
        ADD_INPUTS=True,
    )


# ============================================================================
# Custom Autograd Functions
# ============================================================================

class FusedBaseLinear(torch.autograd.Function):
    """Non-LoRA projections (k/o/mlp/lm_head): fused matmul, train-only backward."""
    @staticmethod
    def forward(ctx, train_x, weight, bias, infer_x):
        B_i, L_i, H = infer_x.shape
        N_i = B_i * L_i
        infer_flat = infer_x.reshape(1, N_i, H)
        combined = torch.cat([infer_flat, train_x], dim=1)
        out = F.linear(combined, weight, bias)
        ctx.save_for_backward(weight)
        O = out.shape[-1]
        infer_base = out[:, :N_i].reshape(B_i, L_i, O)
        train_base = out[:, N_i:]
        return infer_base, train_base

    @staticmethod
    def backward(ctx, grad_infer, grad_train):
        weight, = ctx.saved_tensors
        grad_train_x = grad_train @ weight
        return grad_train_x, None, None, None


class FusedLinearSGMV(torch.autograd.Function):
    """
    Fused base matmul + SGMV LoRA across N request segments.
    Forward: 1 cuBLAS + 2 Triton. Adapter weights accessed via two pointers;
    infer is read directly (no extra copy), ft uses a bf16 dtype-cast buffer.
    Backward: train gradients only (base + ft LoRA); infer skipped.
    """
    # Set by fused_linear / coserving_forward before each .apply() call.
    _a_infer = None      # direct reference to proj.lora_A["infer"].weight (bf16)
    _b_infer = None      # direct reference to proj.lora_B["infer"].weight (bf16)
    _a_ft_buf = None     # bf16 buffer (fp32 ft A cast into it per call)
    _b_ft_buf = None
    _seg_starts = None
    _seg_lens = None
    _seg_adapters = None
    _seg_scalings = None
    _max_seg = 0

    @staticmethod
    def forward(ctx, train_x, infer_x, base_weight, base_bias,
                A_ft_w, B_ft_w, scaling):
        B_i, L_i, H = infer_x.shape
        L_t = train_x.shape[1]
        N_i = B_i * L_i

        infer_flat = infer_x.reshape(1, N_i, H)
        combined = torch.cat([infer_flat, train_x], dim=1)
        base_out = F.linear(combined, base_weight, base_bias)

        mid = sgmv_shrink(
            combined,
            FusedLinearSGMV._a_infer, FusedLinearSGMV._a_ft_buf,
            FusedLinearSGMV._seg_starts, FusedLinearSGMV._seg_lens,
            FusedLinearSGMV._seg_adapters, FusedLinearSGMV._seg_scalings,
            FusedLinearSGMV._max_seg,
        )
        sgmv_expand(
            mid,
            FusedLinearSGMV._b_infer, FusedLinearSGMV._b_ft_buf,
            FusedLinearSGMV._seg_starts, FusedLinearSGMV._seg_lens,
            FusedLinearSGMV._seg_adapters, base_out,
            FusedLinearSGMV._max_seg,
        )

        ctx.save_for_backward(train_x, base_weight, A_ft_w, B_ft_w)
        ctx.scaling = scaling

        O = base_out.shape[-1]
        infer_base = base_out[:, :N_i].reshape(B_i, L_i, O)
        train_base = base_out[:, N_i:]
        return infer_base, train_base

    @staticmethod
    def backward(ctx, grad_infer, grad_train):
        train_x, base_weight, A_ft_w, B_ft_w = ctx.saved_tensors
        scaling = ctx.scaling

        # Base gradient (bf16, train segment only)
        grad_train_x = grad_train @ base_weight

        # LoRA backward in bf16: cast small weight tensors only
        A_bf16 = A_ft_w.bfloat16()
        B_bf16 = B_ft_w.bfloat16()

        train_mid = F.linear(train_x, A_bf16) * scaling   # (1, L_t, R), R=8

        grad_mid = grad_train @ B_bf16                     # (1, L_t, R)
        grad_B = (grad_train.squeeze(0).T @ train_mid.squeeze(0)).float()
        grad_z = grad_mid * scaling
        grad_train_x = grad_train_x + grad_z @ A_bf16
        grad_A = (grad_z.squeeze(0).T @ train_x.squeeze(0)).float()

        return grad_train_x, None, None, None, grad_A, grad_B, None


def fused_linear(proj, infer_x, train_x):
    if hasattr(proj, 'base_layer'):
        base_layer = proj.base_layer
        # Direct references to PEFT Parameter storage — no extra copy of infer
        a_infer = proj.lora_A["infer"].weight   # (R, H) bf16
        b_infer = proj.lora_B["infer"].weight   # (O, R) bf16
        # Tiny bf16 buffer for ft (dtype conversion from fp32 → bf16)
        a_ft_buf, b_ft_buf = _get_ft_bufs(a_infer.shape, b_infer.shape, infer_x.device)
        a_ft_buf.copy_(proj.lora_A["ft"].weight)  # fp32 → bf16 (auto cast)
        b_ft_buf.copy_(proj.lora_B["ft"].weight)

        FusedLinearSGMV._a_infer = a_infer
        FusedLinearSGMV._b_infer = b_infer
        FusedLinearSGMV._a_ft_buf = a_ft_buf
        FusedLinearSGMV._b_ft_buf = b_ft_buf

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


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, cos, sin):
    cos = cos.unsqueeze(2)
    sin = sin.unsqueeze(2)
    return (q * cos + rotate_half(q) * sin), (k * cos + rotate_half(k) * sin)


def attn_forward(attn, infer_h, train_h, infer_rope, train_rope, layer_idx):
    B_i, L_i, _ = infer_h.shape
    L_t = train_h.shape[1]

    infer_q, train_q = fused_linear(attn.q_proj, infer_h, train_h)
    infer_k, train_k = fused_linear(attn.k_proj, infer_h, train_h)
    infer_v, train_v = fused_linear(attn.v_proj, infer_h, train_h)

    # Infer path with per-request KV cache (no_grad)
    with torch.no_grad():
        infer_q = infer_q.view(B_i, L_i, num_heads, head_dim)
        infer_k = infer_k.view(B_i, L_i, num_kv_heads, head_dim)
        infer_v = infer_v.view(B_i, L_i, num_kv_heads, head_dim)
        cos_i, sin_i = infer_rope
        infer_q, infer_k = apply_rope(infer_q, infer_k, cos_i, sin_i)

        if _kv_cache_k[layer_idx] is not None:
            infer_k = torch.cat([_kv_cache_k[layer_idx], infer_k], dim=1)
            infer_v = torch.cat([_kv_cache_v[layer_idx], infer_v], dim=1)
        _kv_cache_k[layer_idx] = infer_k
        _kv_cache_v[layer_idx] = infer_v

        infer_attn = flash_attn_func(infer_q, infer_k, infer_v, causal=True)
        infer_attn = infer_attn.reshape(B_i, L_i, -1)

    # Train path (with grad)
    with torch.enable_grad():
        train_q = train_q.view(1, L_t, num_heads, head_dim)
        train_k = train_k.view(1, L_t, num_kv_heads, head_dim)
        train_v = train_v.view(1, L_t, num_kv_heads, head_dim)
        cos_t, sin_t = train_rope
        train_q, train_k = apply_rope(train_q, train_k, cos_t, sin_t)
        train_attn = flash_attn_func(train_q, train_k, train_v, causal=True)
        train_attn = train_attn.reshape(1, L_t, -1)

    infer_out, train_out = fused_linear(attn.o_proj, infer_attn, train_attn)
    return infer_out, train_out


def mlp_forward(layer, infer_h, train_h):
    mlp = layer.mlp
    infer_n = layer.post_attention_layernorm(infer_h)
    train_n = layer.post_attention_layernorm(train_h)

    infer_gate, train_gate = fused_linear(mlp.gate_proj, infer_n, train_n)
    infer_up,   train_up   = fused_linear(mlp.up_proj,   infer_n, train_n)

    with torch.no_grad():
        infer_act = F.silu(infer_gate) * infer_up
    train_act = F.silu(train_gate) * train_up

    infer_down, train_down = fused_linear(mlp.down_proj, infer_act, train_act)
    return infer_h + infer_down, train_h + train_down


def _build_segments(B_i, L_i, L_t, scaling_infer, scaling_ft, device):
    """Per-segment metadata for this forward (rebuilt per step for dynamic serving)."""
    starts = [i * L_i for i in range(B_i)] + [B_i * L_i]
    lens = [L_i] * B_i + [L_t]
    adapters = [0] * B_i + [1]   # 0 = infer, 1 = ft
    scalings = [scaling_infer] * B_i + [scaling_ft]
    return (
        torch.tensor(starts, dtype=torch.int32, device=device),
        torch.tensor(lens, dtype=torch.int32, device=device),
        torch.tensor(adapters, dtype=torch.int32, device=device),
        torch.tensor(scalings, dtype=torch.float32, device=device),
        max(L_i, L_t),
    )


def coserving_forward(infer_ids, train_ids, infer_pos):
    B_i, L_i = infer_ids.shape
    L_t = train_ids.shape[1]

    sample_proj = base.layers[0].self_attn.q_proj
    scaling_infer = sample_proj.scaling["infer"]
    scaling_ft = sample_proj.scaling["ft"]
    starts, lens, adapters, scalings, max_seg = _build_segments(
        B_i, L_i, L_t, scaling_infer, scaling_ft, infer_ids.device,
    )
    FusedLinearSGMV._seg_starts = starts
    FusedLinearSGMV._seg_lens = lens
    FusedLinearSGMV._seg_adapters = adapters
    FusedLinearSGMV._seg_scalings = scalings
    FusedLinearSGMV._max_seg = max_seg

    infer_h = base.embed_tokens(infer_ids).detach()
    train_h = base.embed_tokens(train_ids)

    device = infer_ids.device
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
    infer_logits, train_logits = fused_linear(lm_head, infer_h, train_h)
    return infer_logits, train_logits


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------

device = input_ids.device
B_i = input_ids.shape[0]  # 2 inference requests
fused_fwd_times = []
bwd_opt_times = []

# Prefill (step 0)
model.train()
train_ids_0 = train_all_ids[0:1]
infer_pos_0 = torch.arange(prompt_len, device=device).unsqueeze(0).expand(B_i, -1)

torch.cuda.synchronize()
t0 = time.perf_counter()
infer_logits, train_logits = coserving_forward(input_ids, train_ids_0, infer_pos_0)
torch.cuda.synchronize()
t1 = time.perf_counter()

next_token = infer_logits[:, -1, :].argmax(dim=-1, keepdim=True)
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

print(f"step   0 | loss={loss.item():.4f} | fused_fwd={t1-t0:.4f}s | bwd+opt={t3-t2:.4f}s (prefill)")

# Decode (steps 1+)
for step in range(1, max_new_tokens):
    model.train()
    train_ids = train_all_ids[step:step+1]
    infer_pos = torch.full((B_i, 1), prompt_len + step - 1, device=device, dtype=torch.long)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    infer_logits, train_logits = coserving_forward(next_token, train_ids, infer_pos)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    next_token = infer_logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated_tokens.append(next_token)

    if (next_token == tokenizer.eos_token_id).all().item():
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
        torch.save(ckpt, "weights_v3_fa_step30.pt")
        print(f"[CHECKPOINT] step 30 weight sum: {sum(v.sum().item() for v in ckpt.values()):.10f}")

    if step % 10 == 0:
        print(f"step {step:3d} | loss={loss.item():.4f} | fused_fwd={t1-t0:.4f}s | bwd+opt={t3-t2:.4f}s")

all_tokens = torch.cat([input_ids] + generated_tokens, dim=-1)
response_1 = tokenizer.decode(all_tokens[0][prompt_len:], skip_special_tokens=True)
response_2 = tokenizer.decode(all_tokens[1][prompt_len:], skip_special_tokens=True)
print(f"\n--- Generated response (request 1) ---\n{response_1}")
print(f"\n--- Generated response (request 2) ---\n{response_2}")

if fused_fwd_times:
    avg_fused = sum(fused_fwd_times) / len(fused_fwd_times)
    avg_bwd = sum(bwd_opt_times) / len(bwd_opt_times)
    print(f"\n--- Timing (after {warmup_steps} warmup, {len(fused_fwd_times)} measured steps) ---")
    print(f"  avg fused_fwd (decode+peft): {avg_fused*1000:.2f} ms")
    print(f"  avg backward+optimizer:      {avg_bwd*1000:.2f} ms")
    print(f"  avg total:                   {(avg_fused+avg_bwd)*1000:.2f} ms")
