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

# Apply two LoRA adapters: one for inference, one for training
lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.0)
model = get_peft_model(model, lora_config, adapter_name="ft")
model.add_adapter("infer", lora_config)
model.set_adapter("ft")  # only ft adapter requires grad
model.print_trainable_parameters()

# Inference prompt
prompt = "Explain what machine learning is in one sentence."
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

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
_kv_cache_k = [None] * num_layers
_kv_cache_v = [None] * num_layers


# --------------------------------------------------------------------------
# Matrix-level fusion with Flash Attention: for each Linear projection,
# cat inference + training along seq dim -> single matmul -> split -> detach
# inference. Non-parametric ops (RoPE, FlashAttn, SiLU) run separately.
# --------------------------------------------------------------------------

def fused_linear(proj, infer_x, train_x):
    """
    Fused base weight load (one HBM read) + separate LoRA adapters.
    For LoRA layers (q_proj, v_proj): base fused, adapters separate.
    For regular layers (k_proj, o_proj, mlp, lm_head): fully fused.
    """
    L_i = infer_x.shape[1]

    if hasattr(proj, 'base_layer'):
        # LoRA layer: fuse base weight, apply separate adapters
        base_out = proj.base_layer(torch.cat([infer_x, train_x], dim=1))
        infer_base = base_out[:, :L_i]
        train_base = base_out[:, L_i:]

        # LoRA weights may differ in dtype from activations; cast to match each adapter
        infer_lora_dtype = proj.lora_A["infer"].weight.dtype
        ft_lora_dtype = proj.lora_A["ft"].weight.dtype
        act_dtype = infer_base.dtype

        # Inference adapter (no dropout for eval)
        infer_lora = proj.lora_B["infer"](proj.lora_A["infer"](infer_x.to(infer_lora_dtype))) * proj.scaling["infer"]
        # Training adapter (with dropout)
        train_lora = proj.lora_B["ft"](proj.lora_A["ft"](proj.lora_dropout["ft"](train_x).to(ft_lora_dtype))) * proj.scaling["ft"]

        return (infer_base + infer_lora.to(act_dtype)).detach(), train_base + train_lora.to(act_dtype)
    else:
        # Regular linear: just fuse
        out = proj(torch.cat([infer_x, train_x], dim=1))
        return out[:, :L_i].detach(), out[:, L_i:]


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, cos, sin):
    """Apply RoPE. q/k shape: (batch, seqlen, nheads, head_dim)"""
    cos = cos.unsqueeze(2)  # (1, seqlen, 1, head_dim)
    sin = sin.unsqueeze(2)  # (1, seqlen, 1, head_dim)
    return (q * cos + rotate_half(q) * sin), (k * cos + rotate_half(k) * sin)


def attn_forward(attn, infer_h, train_h, infer_rope, train_rope, layer_idx):
    L_i, L_t = infer_h.shape[1], train_h.shape[1]
    infer_q, train_q = fused_linear(attn.q_proj, infer_h, train_h)
    infer_k, train_k = fused_linear(attn.k_proj, infer_h, train_h)
    infer_v, train_v = fused_linear(attn.v_proj, infer_h, train_h)

    # Infer path: KV cache
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
fused_fwd_times = []
bwd_opt_times = []

# Prefill (step 0)
model.train()
train_ids_0 = train_all_ids[0:1]
infer_pos_0 = torch.arange(prompt_len, device=device).unsqueeze(0)

torch.cuda.synchronize()
t0 = time.perf_counter()
infer_logits, train_logits = coserving_forward(input_ids, train_ids_0, infer_pos_0)
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

print(f"step   0 | loss={loss.item():.4f} | fused_fwd={t1-t0:.4f}s | bwd+opt={t3-t2:.4f}s (prefill)")

# Decode (steps 1+)
for step in range(1, max_new_tokens):
    model.train()
    train_ids = train_all_ids[step:step+1]
    infer_pos = torch.full((1, 1), prompt_len + step - 1, device=device, dtype=torch.long)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    infer_logits, train_logits = coserving_forward(next_token, train_ids, infer_pos)
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
        torch.save(ckpt, "weights_v1_fa_step30.pt")
        print(f"[CHECKPOINT] step 30 weight sum: {sum(v.sum().item() for v in ckpt.values()):.10f}")

    if step % 10 == 0:
        print(f"step {step:3d} | loss={loss.item():.4f} | fused_fwd={t1-t0:.4f}s | bwd+opt={t3-t2:.4f}s")

all_tokens = torch.cat([input_ids] + generated_tokens, dim=-1)
response = tokenizer.decode(all_tokens[0][prompt_len:], skip_special_tokens=True)
print(f"\n--- Generated response ---\n{response}")

if fused_fwd_times:
    avg_fused = sum(fused_fwd_times) / len(fused_fwd_times)
    avg_bwd = sum(bwd_opt_times) / len(bwd_opt_times)
    print(f"\n--- Timing (after {warmup_steps} warmup steps, {len(fused_fwd_times)} measured steps) ---")
    print(f"  avg fused_fwd (decode+peft): {avg_fused*1000:.2f} ms")
    print(f"  avg backward+optimizer:      {avg_bwd*1000:.2f} ms")
    print(f"  avg total:                   {(avg_fused+avg_bwd)*1000:.2f} ms")
