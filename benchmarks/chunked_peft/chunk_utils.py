import os
import random
import inspect
from collections import OrderedDict
from typing import List, Dict, Tuple, Union, Optional

import torch
import numpy as np
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    DynamicCache,
)
from transformers.utils import (
    is_flash_attn_2_available, 
    is_flash_attn_greater_or_equal,
)
from transformers.modeling_flash_attention_utils import (
    fa_peft_integration_check,
    _upad_input,
    pad_input,
)
from peft import LoraConfig, get_peft_model

if is_flash_attn_2_available():
    from flash_attn.flash_attn_interface import (
        _wrapped_flash_attn_forward,
        _wrapped_flash_attn_backward,
        flash_attn_func, 
        flash_attn_varlen_func,
    )
    _flash_supports_window_size = "window_size" in list(inspect.signature(flash_attn_func).parameters)
    flash_241 = is_flash_attn_greater_or_equal("2.4.1")
    deterministic_g = os.environ.get("FLASH_ATTENTION_DETERMINISTIC", "0") == "1"


class ChunkFlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_softmax,
        start_pos=0, end_pos=-1, layer_idx=-1, past_kv_grads=None,
    ):
        ctx.layer_idx = layer_idx
        ctx.start_pos = start_pos
        ctx.end_pos = end_pos
        ctx.past_kv_grads = past_kv_grads
        
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        head_size_og = q.size(3)
        if head_size_og % 8 != 0:
            q = torch.nn.functional.pad(q, [0, 8 - head_size_og % 8])
            k = torch.nn.functional.pad(k, [0, 8 - head_size_og % 8])
            v = torch.nn.functional.pad(v, [0, 8 - head_size_og % 8])
        out_padded, softmax_lse, S_dmask, rng_state = _wrapped_flash_attn_forward(
            q,
            k,
            v,
            dropout_p,
            softmax_scale,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            return_softmax=return_softmax and dropout_p > 0,
        )
        ctx.save_for_backward(q, k, v, out_padded, softmax_lse, rng_state)
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.softcap = softcap
        ctx.alibi_slopes = alibi_slopes
        ctx.deterministic = deterministic
        out = out_padded[..., :head_size_og]
        return out if not return_softmax else (out, softmax_lse, S_dmask)

    @staticmethod
    def backward(ctx, dout, *args):
        start_pos: int = ctx.start_pos
        end_pos: int = ctx.end_pos
        layer_idx: int = ctx.layer_idx
        past_kv_grads: ChunkCache = ctx.past_kv_grads
        
        q, k, v, out, softmax_lse, rng_state = ctx.saved_tensors
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        head_size_og = dout.size(3)
        dout_padded = dout
        if head_size_og % 8 != 0:
            dout_padded = torch.nn.functional.pad(dout, [0, 8 - head_size_og % 8])
        _wrapped_flash_attn_backward(
            dout_padded,
            q,
            k,
            v,
            out,
            softmax_lse,
            dq,
            dk,
            dv,
            ctx.dropout_p,
            ctx.softmax_scale,
            ctx.causal,
            ctx.window_size[0],
            ctx.window_size[1],
            ctx.softcap,
            ctx.alibi_slopes,
            ctx.deterministic,
            rng_state=rng_state,
        )
        dq = dq[..., : dout.shape[-1]]  # We could have padded the head dimension
        dk = dk[..., : dout.shape[-1]]
        dv = dv[..., : dout.shape[-1]]

        dk, dv = past_kv_grads.update_grads(dk, dv, start_pos, end_pos, layer_idx)

        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None, None, None

def chunk_flash_attn_func(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    softcap=0.0, # 0.0 means deactivated
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    start_pos=0, end_pos=-1, layer_idx=-1, past_kv_grads=None,
):
    return ChunkFlashAttnFunc.apply(
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        start_pos, end_pos, layer_idx, past_kv_grads,
    )


def _chunk_flash_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor,
    query_length: int,
    is_causal: bool,
    dropout: float = 0.0,
    position_ids: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    sliding_window: Optional[int] = None,
    use_top_left_mask: bool = False,
    softcap: Optional[float] = None,
    deterministic: bool = None,
    cu_seq_lens_q: Optional[torch.LongTensor] = None,
    cu_seq_lens_k: Optional[torch.LongTensor] = None,
    max_length_q: Optional[int] = None,
    max_length_k: Optional[int] = None,
    target_dtype: Optional[torch.dtype] = None,
    start_pos: int = 0, end_pos: int = -1, layer_idx: int = -1, past_kv_grads: "ChunkCache" = None,
):
    """
    Modified from HuggingFace transformers v4.47.0 src/transformers/modeling_flash_attention_utils.py _flash_attention_forward
    """
    if not use_top_left_mask:
        causal = is_causal
    else:
        # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in transformers.models.llama.modeling_llama.LlamaFlashAttention2.__init__.
        causal = is_causal and query_length != 1

    # Assuming 4D tensors, key_states.shape[1] is the key/value sequence length (source length).
    use_sliding_windows = (
        _flash_supports_window_size and sliding_window is not None and key_states.shape[1] > sliding_window
    )
    flash_kwargs = {"window_size": (sliding_window, sliding_window)} if use_sliding_windows else {}

    if flash_241:
        if deterministic is None:
            deterministic = deterministic_g
        flash_kwargs["deterministic"] = deterministic

    if softcap is not None:
        flash_kwargs["softcap"] = softcap

    # PEFT possibly silently casts tensors to fp32, this potentially reconverts to correct dtype or is a no op
    query_states, key_states, value_states = fa_peft_integration_check(
        query_states, key_states, value_states, target_dtype
    )

    # Contains at least one padding token in the sequence
    if attention_mask is not None:
        batch_size = query_states.shape[0]
        query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = _upad_input(
            query_states, key_states, value_states, attention_mask, query_length
        )
        cu_seqlens_q, cu_seqlens_k = cu_seq_lens
        max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

        attn_output_unpad = flash_attn_varlen_func(
            query_states,
            key_states,
            value_states,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_in_batch_q,
            max_seqlen_k=max_seqlen_in_batch_k,
            dropout_p=dropout,
            softmax_scale=softmax_scale,
            causal=causal,
            **flash_kwargs,
        )
        attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)

    # If position_ids is provided and check all examples do not contain only 1 sequence, If tensor in increasing
    # then we probably have one sequence, otherwise it is packed. Additionally check we are in pre-fill/training stage.
    # Use `flash_attn_varlen_func` to prevent cross-example attention and also allow padding free approach
    elif position_ids is not None and (
        max_length_q is not None or (query_length != 1 and not (torch.diff(position_ids, dim=-1) >= 0).all())
    ):
        batch_size = query_states.size(0)

        if cu_seq_lens_q is None or cu_seq_lens_k is None:
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = (
                prepare_fa2_from_position_ids(query_states, key_states, value_states, position_ids)
            )

            cu_seq_lens_q, cu_seq_lens_k = cu_seq_lens
            max_length_q, max_length_k = max_seq_lens

        else:
            query_states = query_states.reshape(-1, query_states.size(-2), query_states.size(-1))
            key_states = key_states.reshape(-1, key_states.size(-2), key_states.size(-1))
            value_states = value_states.reshape(-1, value_states.size(-2), value_states.size(-1))

        attn_output = flash_attn_varlen_func(
            query_states,
            key_states,
            value_states,
            cu_seqlens_q=cu_seq_lens_q,
            cu_seqlens_k=cu_seq_lens_k,
            max_seqlen_q=max_length_q,
            max_seqlen_k=max_length_k,
            dropout_p=dropout,
            softmax_scale=softmax_scale,
            causal=causal,
            **flash_kwargs,
        )

        attn_output = attn_output.view(batch_size, -1, attn_output.size(-2), attn_output.size(-1))

    else:
        attn_output = chunk_flash_attn_func(
            query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal, 
            start_pos=start_pos, end_pos=end_pos, layer_idx=layer_idx, past_kv_grads=past_kv_grads, **flash_kwargs
        )

    return attn_output



class ChunkSDPA(torch.autograd.Function):
    @staticmethod
    def forward(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, 
                start_pos=0, end_pos=-1, layer_idx=-1, past_kv_grads=None):
        output = torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask, dropout_p, is_causal)
        return output
    
    @staticmethod
    def setup_context(ctx, inputs, output):
        query, key, value, attn_mask, dropout_p, is_causal, start_pos, end_pos, layer_idx, past_kv_grads = inputs
        ctx.save_for_backward(query, key, value, attn_mask)
        ctx.dropout_p = dropout_p
        ctx.is_causal = is_causal
        ctx.layer_idx = layer_idx
        ctx.start_pos = start_pos
        ctx.end_pos = end_pos
        ctx.past_kv_grads = past_kv_grads
    
    @staticmethod
    def backward(ctx, grad_output):
        # read range start & range end
        start_pos: int = ctx.start_pos
        end_pos: int = ctx.end_pos
        # read layer idx
        layer_idx: int = ctx.layer_idx
        
        # print(f"start: {start_pos}; end: {end_pos}; layer_idx: {layer_idx}")
        
        # add grad back to cache
        past_kv_grads: ChunkCache = ctx.past_kv_grads

        query, key, value, attn_mask = ctx.saved_tensors
        dropout_p = ctx.dropout_p
        is_causal = ctx.is_causal
        
        # compute grads manually
        grad_query = grad_key = grad_value = None
        
        # dQ, dK, dV (here assumes torch handles it)
        if query.requires_grad or key.requires_grad or value.requires_grad:
            with torch.enable_grad():
                query = query.detach().requires_grad_()
                key = key.detach().requires_grad_()
                value = value.detach().requires_grad_()
                
                # recompute output for gradient calculation
                # FIXME: consider use ctx to save outputs to avoid re-computation
                output = torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask, dropout_p, is_causal)
                
                # torch.autograd.grad to compute gradients
                grads = torch.autograd.grad(outputs=output, 
                                            inputs=(query, key, value), 
                                            grad_outputs=grad_output, 
                                            allow_unused=True)
                grad_query, grad_key, grad_value = grads

                # return corresponding range of gradients
                grad_key, grad_value = past_kv_grads.update_grads(grad_key, grad_value, start_pos, end_pos, layer_idx)

        return grad_query, grad_key, grad_value, None, None, None, None, None, None, None


def chunk_sdpa(query_states: torch.Tensor,
               key_states: torch.Tensor,
               value_states: torch.Tensor,
               attn_mask: torch.Tensor,
               dropout_p: float,
               is_causal: bool,
               start_pos: int,
               end_pos: int,
               layer_idx: int,
               past_kv_grads: "ChunkCache"):
    return ChunkSDPA.apply(
        query_states, 
        key_states, 
        value_states, 
        attn_mask, 
        dropout_p, 
        is_causal,
        start_pos,
        end_pos,
        layer_idx,
        past_kv_grads,
    )


class ChunkCache(DynamicCache):
    def __init__(self, attn_impl: str = "flash_attention_2") -> None:
        super().__init__()
        self.key_grads_cache = []
        self.value_grads_cache = []
        self.max_layer_idx = None
        self.attn_impl = attn_impl
    
    def detach_kv_cache(self):
        new_key_cache = []
        new_value_cache = []
        for key_tensor in self.key_cache:
            new_key_cache.append(key_tensor.detach().to(device=key_tensor.device).requires_grad_())
        for value_tensor in self.value_cache:
            new_value_cache.append(value_tensor.detach().to(device=value_tensor.device).requires_grad_())
        self.key_cache = new_key_cache
        self.value_cache = new_value_cache

    def update_grads(self, key_grad, value_grad, start_pos, end_pos, layer_idx):
        # update max layer idx
        if self.max_layer_idx is None:
            self.max_layer_idx = layer_idx

        # update the cache
        cur_layer_idx: int = self.max_layer_idx - layer_idx

        # print(f"cur_layer_idx: {cur_layer_idx}; len(key_grads_cache): {len(self.key_grads_cache)}")

        if len(self.key_grads_cache) <= cur_layer_idx:
            # there may be skipped layers, fill them with empty lists
            for _ in range(len(self.key_grads_cache), cur_layer_idx):
                self.key_grads_cache.append([])
                self.value_grads_cache.append([])
            self.key_grads_cache.append(key_grad.detach())
            self.value_grads_cache.append(value_grad.detach())
        elif len(self.key_grads_cache[cur_layer_idx]) == 0:  # fills previously skipped layers; checking for tensor causes errors
            self.key_grads_cache[cur_layer_idx] = key_grad.detach()
            self.value_grads_cache[cur_layer_idx] = value_grad.detach()
        else:
            # print(f"key_grads_cache[{cur_layer_idx}]: {self.key_grads_cache[cur_layer_idx].shape}")
            # print(f"key_grad: {key_grad.shape}")
            if self.attn_impl == "flash_attention_2":
                self.key_grads_cache[cur_layer_idx] = self.key_grads_cache[cur_layer_idx][:,:end_pos,:,:] + key_grad.detach()
                self.value_grads_cache[cur_layer_idx] = self.value_grads_cache[cur_layer_idx][:,:end_pos,:,:] + value_grad.detach()
            else:
                self.key_grads_cache[cur_layer_idx] = self.key_grads_cache[cur_layer_idx][:,:,:end_pos,:] + key_grad.detach()
                self.value_grads_cache[cur_layer_idx] = self.value_grads_cache[cur_layer_idx][:,:,:end_pos,:] + value_grad.detach()
        
        if self.attn_impl == "flash_attention_2":
            valid_key_grads = torch.cat([torch.zeros_like(self.key_grads_cache[cur_layer_idx])[:,:start_pos,:,:], 
                                        self.key_grads_cache[cur_layer_idx][:,start_pos:end_pos:,:,:]], dim=1)
            valid_value_grads = torch.cat([torch.zeros_like(self.value_grads_cache[cur_layer_idx])[:,:start_pos,:,:], 
                                          self.value_grads_cache[cur_layer_idx][:,start_pos:end_pos:,:,:]], dim=1)
        else:
            valid_key_grads = torch.cat([torch.zeros_like(self.key_grads_cache[cur_layer_idx])[:,:,:start_pos,:], 
                                        self.key_grads_cache[cur_layer_idx][:,:,start_pos:end_pos:,:]], dim=-2)
            valid_value_grads = torch.cat([torch.zeros_like(self.value_grads_cache[cur_layer_idx])[:,:,:start_pos,:], 
                                        self.value_grads_cache[cur_layer_idx][:,:,start_pos:end_pos:,:]], dim=-2)
        # print(f"cur_layer_idx: {cur_layer_idx}; len(key_grads_cache): {len(self.key_grads_cache)}")
        return valid_key_grads, valid_value_grads


class Printer:
    def __init__(self) -> None:
        self.PRINT_OUT = True
        torch.set_printoptions(precision=16, sci_mode=False)

    def open(self) -> None:
        self.PRINT_OUT = True

    def close(self) -> None:
        self.PRINT_OUT = False

    def print(self, context: str) -> None:
        if self.PRINT_OUT:
            print(context)

printer = Printer()

INVALID_TARGET: int = -100

def fix_determinism(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

def fixed_cross_entropy(source, target, num_items_in_batch: int = None, ignore_index: int = -100, **kwargs):
    reduction = "sum" if num_items_in_batch is not None else "mean"
    loss = torch.nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
    if reduction == "sum":
        loss = loss / num_items_in_batch
    return loss

def get_hf_model(
    model_name: str,
    pin_memory: Union[str, bool],
    quant_bits: int,
    quant_group_size: int,
    cache_dir: str,
    dtype: torch.dtype = torch.bfloat16,
    attn_impl: str = "flash_attn",
    lora_r: int = 64,
    lora_alpha: int = 32,
    lora_dropout: float = 0.0,
    target_modules: List[str] = ["q_proj", "v_proj"],
) -> torch.nn.Module:
    # FIXME: for numerical verification based on float64 compatible math sdp
    torch.backends.cuda.enable_math_sdp(enabled=False)
    torch.backends.cuda.enable_flash_sdp(enabled=False)
    torch.backends.cuda.enable_mem_efficient_sdp(enabled=False)
    if attn_impl == "math":
        torch.backends.cuda.enable_math_sdp(enabled=True)
    elif attn_impl == "mem_efficient":
        torch.backends.cuda.enable_mem_efficient_sdp(enabled=True)
    elif attn_impl == "flash":
        print(f"im here")
        torch.backends.cuda.enable_flash_sdp(enabled=True)

    config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
    if attn_impl in ["math", "flash", "mem_efficient"]:
        config._attn_implementation = "sdpa"
    else:
        config._attn_implementation = attn_impl
    
    # config.num_hidden_layers = 4 # FIXME: for visualization
    
    pin_memory = bool(pin_memory)
    if quant_bits == 4:
        raise NotImplementedError()

    base_model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                      config=config, 
                                                      cache_dir=cache_dir, 
                                                      torch_dtype=dtype)
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_config)
    model.to(torch.cuda.current_device())

    return model

def copy_models(src_model: torch.nn.Module, tgt_model: torch.nn.Module) -> None:
    printer.print("synchronizing initial parameters...")
    tgt_model.load_state_dict(src_model.state_dict())

def check_two_same_models(src_model: torch.nn.Module, tgt_model: torch.nn.Module) -> None:
    printer.print("checking if the two models are the same...")
    for w, c in zip(list(src_model.parameters()), list(tgt_model.parameters())):
        assert (w.data.cpu() == c.data.cpu()).all()

def create_chunk_model(model_name: str,
                       pin_memory: bool,
                       quant_bits: int,
                       quant_group_size: int,
                       cache_dir: str,
                       local_ranks: Union[int, List[int]] = 0,
                       dtype: torch.dtype = torch.bfloat16,
                       attn_impl: str = "flash_attn",
                       lora_r: int = 64,
                       lora_alpha: int = 32,
                       lora_dropout: float = 0.0,
                       target_modules: List[str] = ["q_proj", "v_proj"],
                       ) -> Union[torch.nn.Module, List[torch.nn.Module]]:
    if not isinstance(local_ranks, list):
        local_ranks: List[int] = [ local_ranks ]
    assert len(local_ranks) > 0
    models: List[torch.nn.Module] = []
    for idx, local_rank in enumerate(local_ranks):
        torch.cuda.set_device(local_rank)
        printer.print(f"loading model {idx}...")
        # for comparison, always set the first model as eager impl [whole fwd & bwd]
        cur_attn_impl: str = "eager" if len(local_ranks) > 1 and idx == 0 else attn_impl
        model: torch.nn.Module = get_hf_model(
            model_name,
            pin_memory,
            quant_bits,
            quant_group_size,
            cache_dir,
            dtype=dtype,
            attn_impl=cur_attn_impl,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
        )
        if (idx > 0):
            copy_models(models[0], model)
            check_two_same_models(models[0], model)
        models.append(model)
    torch.cuda.set_device(local_ranks[0])
    return models[0] if len(models) == 1 else models

def get_learnable_param_grad(model: torch.nn.Module) -> OrderedDict:
    res_dict = OrderedDict()
    for name, param in model.named_parameters():
        if param.grad is not None:
            res_dict[name] = param.grad
    return res_dict

def copy_input(inputs: Union[List[int], Dict], local_rank: int) -> Union[List[int], Dict]:
    with torch.no_grad():
        new_inputs = inputs
        new_inputs.input_ids = inputs.input_ids.clone().detach().to(device=local_rank)
        new_inputs.attention_mask = inputs.attention_mask.clone().detach().to(device=local_rank)
        new_inputs.labels = inputs.labels.clone().detach().to(device=local_rank)
    return new_inputs

def get_dtype(dtype_str: str) -> torch.dtype:
    if dtype_str == "fp16":
        return torch.float16
    elif dtype_str == "bf16":
        return torch.bfloat16
    elif dtype_str == "fp32":
        return torch.float32
    elif dtype_str == "fp64":
        return torch.float64
    else:
        raise NotImplementedError()

def add_print_backward_hooks(model: torch.nn.Module):
    def backward_hook_fn(module: torch.nn.Module, 
                         grad_in: torch.Tensor, 
                         grad_out: torch.Tensor):
        if grad_in[0] is not None:
            printer.print(f"module: {module._get_name()}; " + \
                          f"grad_in shape: {grad_in[0].shape}; " + \
                          f"grad_out shape: {grad_out[0].shape}")
        else:
            printer.print(f"module: {module._get_name()}; " + \
                          f"grad_in shape: {grad_in[1].shape}; " + \
                          f"grad_out shape: {grad_out[0].shape}")
    
    def register_backward_hooks(model: torch.nn.Module):
        modules = model.named_modules()
        for name, module in modules:
            printer.print(f"add backward hook for module: {name}")
            module.register_backward_hook(backward_hook_fn)
    
    register_backward_hooks(model)
