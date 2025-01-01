import os
from typing import Tuple, Any, Optional, Tuple, Dict

import torch
from torchtune.utils._import_guard import _SUPPORTS_FLEX_ATTENTION
if _SUPPORTS_FLEX_ATTENTION:
    from torch.nn.attention.flex_attention import BlockMask
else:
    BlockMask = torch.Tensor

from benchmarks.chunked_peft.cache_utils import DynamicCache


### cross entropy
def fixed_cross_entropy(source, target, num_items_in_batch: int = None, ignore_index: int = -100, **kwargs):
    reduction = "sum" if num_items_in_batch is not None else "mean"
    loss = torch.nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
    if reduction == "sum":
        loss = loss / num_items_in_batch
    return loss

### casual mask for chunked peft
def prepare_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
):
    """
    Creates a causal 4D mask of shape `(batch_size, query_length, key_value_length)` from a 2D mask of shape
    `(batch_size, key_value_length)`. 
    Modified from https://github.com/huggingface/transformers/blob/v4.47.0/src/transformers/models/mistral/modeling_mistral.py
    """
    
    min_dtype = torch.finfo(dtype).min
    
    causal_mask = torch.full(
        (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
    )
    
    diagonal_attend_mask = torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
    causal_mask *= diagonal_attend_mask
    
    causal_mask = causal_mask[None, :, :].expand(batch_size, -1, -1)
    
    if attention_mask is not None:
        causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
        if attention_mask.shape[-1] > target_length:
            attention_mask = attention_mask[:, :target_length]
        mask_length = attention_mask.shape[-1]
        padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
        padding_mask = padding_mask == 0
        causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
            padding_mask, min_dtype
        )
    
    return causal_mask



### sdpa mode

def set_sdpa_mode(attn_impl: str = "mem_efficient"):
    torch.backends.cuda.enable_math_sdp(enabled=False)
    torch.backends.cuda.enable_flash_sdp(enabled=False)
    torch.backends.cuda.enable_mem_efficient_sdp(enabled=False)
    if torch.__version__ >= "2.5.0":
        torch.backends.cuda.enable_cudnn_sdp(enabled=False)
    
    if attn_impl == "math":
        torch.backends.cuda.enable_math_sdp(enabled=True)
    elif attn_impl == "mem_efficient":
        torch.backends.cuda.enable_mem_efficient_sdp(enabled=True)
    elif attn_impl == "flash":
        torch.backends.cuda.enable_flash_sdp(enabled=True)
    elif attn_impl == "cudnn":
        assert torch.__version__ >= "2.5.0", "Chunked PEFT w/ SDPA cuDNN is only supported in torch 2.5.0+."
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        torch.backends.cuda.enable_cudnn_sdp(enabled=True)


### batch_to_device

def batch_to_device(batch: dict, device: torch.device) -> None:
    """Function that takes a dictionary (or nested dictionary) of tensors and sets them
    all to the same device. This utility is intended to be used for batches of data to be
    moved to device, the update is inplace.

    Args:
        batch (dict): dict of Tensors or more nested dicts of tensors.
        device (torch.device): torch device to move the tensor's too

    Raises:
        AttributeError: if batch dict contains anything other than tensors
    """
    for k, v in batch.items():
        if isinstance(v, dict):
            batch_to_device(v, device)
        elif isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
        elif _SUPPORTS_FLEX_ATTENTION and isinstance(v, BlockMask):
            batch[k] = v.to(device)
        elif isinstance(v, ChunkCache):
            batch[k] = v.to(device)
        else:
            raise ValueError(
                f"""To use batch_to_device, all elements in the batch must be a dict or Tensor.
Got key "{k}" with value of type {type(v)}"""
            )


### Chunked SDPA

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Copied from https://github.com/huggingface/transformers/blob/v4.47.0/src/transformers/models/llama/modeling_llama.py
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class ChunkSDPA(torch.autograd.Function):
    @staticmethod
    def forward(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, 
                start_pos=0, end_pos=-1, layer_idx=-1, past_kv_grads=None):
        output = torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask, dropout_p, is_causal)
        return output
    
    @staticmethod
    def setup_context(ctx, inputs, output):
        (
            query, 
            key, 
            value, 
            attn_mask, 
            dropout_p, 
            is_causal, 
            start_pos, end_pos, layer_idx, past_kv_grads
        ) = inputs

        ctx.save_for_backward(query, attn_mask)
        ctx.dropout_p = dropout_p
        ctx.is_causal = is_causal
        ctx.layer_idx = layer_idx
        ctx.start_pos = start_pos
        ctx.end_pos   = end_pos
        ctx.past_kv_grads = past_kv_grads
    
    @staticmethod
    def backward(ctx, grad_output):
        # read range start & range end
        start_pos: int = ctx.start_pos
        end_pos:   int = ctx.end_pos
        layer_idx: int = ctx.layer_idx
        
        # add grad back to cache
        past_kv_grads: ChunkCache = ctx.past_kv_grads

        query, attn_mask = ctx.saved_tensors
        
        key = past_kv_grads.key_cache[layer_idx][:,:,:end_pos,:].requires_grad_().contiguous()
        value = past_kv_grads.value_cache[layer_idx][:,:,:end_pos,:].requires_grad_().contiguous()

        dropout_p = ctx.dropout_p
        is_causal = ctx.is_causal
        
        # compute grads manually
        grad_query = grad_key = grad_value = None
        if query.requires_grad or key.requires_grad or value.requires_grad:
            with torch.enable_grad():
                query = query.detach().requires_grad_()
                key   = key.detach().requires_grad_()
                value = value.detach().requires_grad_()
                
                # recompute output for gradient calculation
                output = torch.nn.functional.scaled_dot_product_attention(
                    query, 
                    key, 
                    value, 
                    attn_mask, 
                    dropout_p, 
                    is_causal
                )
                
                # torch.autograd.grad to compute gradients
                grads = torch.autograd.grad(outputs=output, 
                                            inputs=(query, key, value), 
                                            grad_outputs=grad_output, 
                                            allow_unused=True)
                grad_query, grad_key, grad_value = grads

                # update prev range grads & return current range of grads
                grad_key, grad_value = past_kv_grads.update_grads(grad_key, grad_value, start_pos, end_pos, layer_idx)

        return grad_query, grad_key, grad_value, None, None, None, None, None, None, None


def chunk_sdpa(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    mask: torch.Tensor,
    dropout_p: float,
    is_causal: bool,
    start_pos: int,
    end_pos: int,
    layer_idx: int,
    past_kv_grads: "ChunkCache",
):
    if mask is not None:
        mask = mask[:, None, :, :]

    return ChunkSDPA.apply(
        query_states, 
        key_states, 
        value_states, 
        mask, 
        dropout_p, 
        is_causal,
        start_pos,
        end_pos,
        layer_idx,
        past_kv_grads,
    )


### ChunkCache w/ key, value & key_grad, value_grad
class ChunkCache(DynamicCache):
    def __init__(self, attn_impl: str = "mem_efficient") -> None:
        super().__init__()
        self.key_grads_cache   = []
        self.value_grads_cache = []
        self.max_layer_idx = None
        self.attn_impl = attn_impl

    def detach_kv_cache(self):
        new_key_cache   = []
        new_value_cache = []
        for key_tensor in self.key_cache:
            new_key_cache.append(key_tensor.detach().to(device=key_tensor.device))
        for value_tensor in self.value_cache:
            new_value_cache.append(value_tensor.detach().to(device=value_tensor.device))
        del self.key_cache
        del self.value_cache
        self.key_cache = new_key_cache
        self.value_cache = new_value_cache

    def reset(self):
        del self.key_cache
        del self.value_cache
        del self.key_grads_cache
        del self.value_grads_cache
        self.key_cache   = []
        self.value_cache = []
        self.key_grads_cache   = []
        self.value_grads_cache = []

    def update_key_grads(self, key_grad, start_pos, end_pos, layer_idx):
        # update max layer idx
        if self.max_layer_idx is None:
            self.max_layer_idx = layer_idx

        # update the cache
        cur_layer_idx: int = self.max_layer_idx - layer_idx

        valid_key_grads = None
        update_key_grads = []
        if key_grad is not None:
            key_grad = key_grad.detach()
            valid_key_grads  = key_grad[:,start_pos:end_pos,:,:] if self.attn_impl == "flash_attention_2" \
                          else key_grad[:,:,start_pos:end_pos,:]
            update_key_grads = key_grad[:,:start_pos,:,:] if self.attn_impl == "flash_attention_2" \
                          else key_grad[:,:,:start_pos,:]

        if len(self.key_grads_cache) <= cur_layer_idx:
            # there may be skipped layers, fill them with empty lists
            for _ in range(len(self.key_grads_cache), cur_layer_idx):
                self.key_grads_cache.append([])
            self.key_grads_cache.append(update_key_grads)
        elif len(self.key_grads_cache[cur_layer_idx]) == 0:  # fills previously skipped layers; checking for tensor causes errors
            self.key_grads_cache[cur_layer_idx] = update_key_grads
        else:
            if key_grad is not None:
                accum_key_grads = self.key_grads_cache[cur_layer_idx][:,:start_pos,:,:] if self.attn_impl == "flash_attention_2" \
                             else self.key_grads_cache[cur_layer_idx][:,:,:start_pos,:]
                accum_valid_key_grads = self.key_grads_cache[cur_layer_idx][:,start_pos:end_pos,:,:] if self.attn_impl == "flash_attention_2" \
                                   else self.key_grads_cache[cur_layer_idx][:,:,start_pos:end_pos,:]
                valid_key_grads += accum_valid_key_grads
                self.key_grads_cache[cur_layer_idx] = accum_key_grads + update_key_grads
            else:
                self.key_grads_cache[cur_layer_idx] = []
        
        if key_grad is not None:
            padding_zeros = torch.zeros_like(key_grad[:,:start_pos,:,:] if self.attn_impl == "flash_attention_2" \
                                        else key_grad[:,:,:start_pos,:])
            valid_key_grads = torch.cat([padding_zeros, valid_key_grads], dim=(1 if self.attn_impl == "flash_attention_2" else 2))
        return valid_key_grads
        
    def update_value_grads(self, value_grad, start_pos, end_pos, layer_idx):
        # update max layer idx
        if self.max_layer_idx is None:
            self.max_layer_idx = layer_idx

        # update the cache
        cur_layer_idx: int = self.max_layer_idx - layer_idx

        valid_value_grads = None
        update_value_grads = []
        if value_grad is not None:
            value_grad = value_grad.detach()
            valid_value_grads  = value_grad[:,start_pos:end_pos,:,:] if self.attn_impl == "flash_attention_2" \
                            else value_grad[:,:,start_pos:end_pos,:]
            update_value_grads = value_grad[:,:start_pos,:,:] if self.attn_impl == "flash_attention_2" \
                            else value_grad[:,:,:start_pos,:]

        if len(self.value_grads_cache) <= cur_layer_idx:
            # there may be skipped layers, fill them with empty lists
            for _ in range(len(self.value_grads_cache), cur_layer_idx):
                self.value_grads_cache.append([])
            self.value_grads_cache.append(update_value_grads)
        elif len(self.value_grads_cache[cur_layer_idx]) == 0:  # fills previously skipped layers; checking for tensor causes errors
            self.value_grads_cache[cur_layer_idx] = update_value_grads
        else:
            if value_grad is not None:
                accum_value_grads = self.value_grads_cache[cur_layer_idx][:,:start_pos,:,:] if self.attn_impl == "flash_attention_2" \
                               else self.value_grads_cache[cur_layer_idx][:,:,:start_pos,:]
                accum_valid_value_grads = self.value_grads_cache[cur_layer_idx][:,start_pos:end_pos,:,:] if self.attn_impl == "flash_attention_2" \
                                     else self.value_grads_cache[cur_layer_idx][:,:,start_pos:end_pos,:]
                valid_value_grads += accum_valid_value_grads
                self.value_grads_cache[cur_layer_idx] = accum_value_grads + update_value_grads
            else:
                self.value_grads_cache[cur_layer_idx] = []
        if value_grad is not None:
            padding_zeros = torch.zeros_like(value_grad[:,:start_pos,:,:] if self.attn_impl == "flash_attention_2" \
                                        else value_grad[:,:,:start_pos,:])
            valid_value_grads = torch.cat([padding_zeros, valid_value_grads], dim=(1 if self.attn_impl == "flash_attention_2" else 2))
        
        return valid_value_grads

    def update_grads(self, key_grad, value_grad, start_pos, end_pos, layer_idx):
        valid_key_grads   = self.update_key_grads(key_grad, start_pos, end_pos, layer_idx)
        valid_value_grads = self.update_value_grads(value_grad, start_pos, end_pos, layer_idx)
        return valid_key_grads, valid_value_grads
