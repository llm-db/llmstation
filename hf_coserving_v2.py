import time
import torch
import torch.nn.functional as F
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

# Two inference requests share the prompt and the infer adapter; gather-BMM
# batches them into the (B_i, L_i) inference sub-tensor of the fused forward.
prompt = "Explain what machine learning is in one sentence."
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device).repeat(2, 1)  # (2, L)

max_new_tokens = 128
warmup_steps = 3

# Pre-tokenize training samples from Alpaca, seq_len=200
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

# Per-step adapter list (decided per decode step, varies in real serving)
GBMM_STEP_ADAPTERS = None


# --------------------------------------------------------------------------
# Custom autograd: fused forward (one HBM weight read), train-only backward.
# Same as v1 fusion but now with B_i inference requests in the concat: infer
# is flattened (B_i, L_i, H) -> (1, B_i*L_i, H) before cat with train.
# Backward only computes dL/d(train_x) = grad_train @ W, matching v0 cost.
# --------------------------------------------------------------------------

class FusedBaseLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, train_x, weight, bias, infer_x):
        B_i, L_i, H = infer_x.shape
        N_i = B_i * L_i
        infer_flat = infer_x.reshape(1, N_i, H)
        combined = torch.cat([infer_flat, train_x], dim=1)
        out = F.linear(combined, weight, bias)
        ctx.save_for_backward(weight)
        ctx.N_i = N_i
        O = out.shape[-1]
        infer_base = out[:, :N_i].reshape(B_i, L_i, O)
        train_base = out[:, N_i:]
        return infer_base, train_base

    @staticmethod
    def backward(ctx, grad_infer, grad_train):
        weight, = ctx.saved_tensors
        grad_train_x = grad_train @ weight
        return grad_train_x, None, None, None


def fused_linear(proj, infer_x, train_x):
    """
    Fused forward (one HBM weight read) + train-only backward.
    Inference LoRA uses gather-BMM across per-slot adapters; train uses ft.
    """
    if hasattr(proj, 'base_layer'):
        base = proj.base_layer
        infer_base, train_base = FusedBaseLinear.apply(
            train_x, base.weight, base.bias, infer_x
        )
        act_dtype = infer_base.dtype

        # Infer LoRA: gather-BMM, stacked on the fly per step's adapter list
        adapters = GBMM_STEP_ADAPTERS
        A_stack = torch.stack([proj.lora_A[a].weight for a in adapters], dim=0)
        B_stack = torch.stack([proj.lora_B[a].weight for a in adapters], dim=0)
        scaling = proj.scaling[adapters[0]]
        infer_x_lora = infer_x.to(A_stack.dtype)
        mid = torch.bmm(infer_x_lora, A_stack.transpose(1, 2))
        infer_lora = torch.bmm(mid, B_stack.transpose(1, 2)) * scaling

        # Train LoRA: ft adapter
        ft_lora_dtype = proj.lora_A["ft"].weight.dtype
        train_lora = proj.lora_B["ft"](proj.lora_A["ft"](proj.lora_dropout["ft"](train_x).to(ft_lora_dtype))) * proj.scaling["ft"]

        return (infer_base + infer_lora.to(act_dtype)).detach(), train_base + train_lora.to(act_dtype)
    else:
        # Regular linear: fused forward, train-only backward
        infer_out, train_out = FusedBaseLinear.apply(
            train_x, proj.weight, proj.bias, infer_x
        )
        return infer_out.detach(), train_out


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, cos, sin):
    """Apply RoPE. q/k shape: (B, L, nheads, head_dim); cos/sin: (B, L, head_dim)."""
    cos = cos.unsqueeze(2)
    sin = sin.unsqueeze(2)
    return (q * cos + rotate_half(q) * sin), (k * cos + rotate_half(k) * sin)


def attn_forward(attn, infer_h, train_h, infer_rope, train_rope, layer_idx):
    B_i, L_i, _ = infer_h.shape
    L_t = train_h.shape[1]
    infer_q, train_q = fused_linear(attn.q_proj, infer_h, train_h)
    infer_k, train_k = fused_linear(attn.k_proj, infer_h, train_h)
    infer_v, train_v = fused_linear(attn.v_proj, infer_h, train_h)

    # Infer path: per-request KV cache (batch B_i)
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

    # Train path: no KV cache
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
    """Fused gate/up/down projections, separate activation."""
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


def coserving_forward(infer_ids, train_ids, infer_pos):
    # infer_ids: (B_i, L_i), train_ids: (1, L_t), infer_pos: (B_i, L_i)
    infer_h = base.embed_tokens(infer_ids).detach()
    train_h = base.embed_tokens(train_ids)

    L_t = train_ids.shape[1]
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

GBMM_STEP_ADAPTERS = ["infer"] * B_i
torch.cuda.synchronize()
t0 = time.perf_counter()
infer_logits, train_logits = coserving_forward(input_ids, train_ids_0, infer_pos_0)
torch.cuda.synchronize()
t1 = time.perf_counter()
GBMM_STEP_ADAPTERS = None

next_token = infer_logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (B_i, 1)
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
    step_adapters = ["infer"] * B_i  # per-step list (could vary in real serving)

    GBMM_STEP_ADAPTERS = step_adapters
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    infer_logits, train_logits = coserving_forward(next_token, train_ids, infer_pos)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    GBMM_STEP_ADAPTERS = None

    next_token = infer_logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (B_i, 1)
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
        torch.save(ckpt, "weights_v2_fa_step30.pt")
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
    print(f"\n--- Timing (after {warmup_steps} warmup steps, {len(fused_fwd_times)} measured steps) ---")
    print(f"  avg fused_fwd (decode+peft): {avg_fused*1000:.2f} ms")
    print(f"  avg backward+optimizer:      {avg_bwd*1000:.2f} ms")
    print(f"  avg total:                   {(avg_fused+avg_bwd)*1000:.2f} ms")
