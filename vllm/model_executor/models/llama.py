# coding=utf-8
# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only LLaMA model compatible with HuggingFace weights."""
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
from torch import nn
from transformers import LlamaConfig

from vllm.attention import Attention, AttentionMetadata
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, LoRAConfig
from vllm.distributed import (get_pp_group, get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size)
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
    get_compressed_tensors_cache_scale)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import Sampler, SamplerOutput
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, kv_cache_scales_loader, maybe_remap_kv_scale_name)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors
from vllm.utils import is_hip

from .interfaces import SupportsLoRA, SupportsPP
from .utils import (AutoWeightsLoader, PPMissingLayer, is_pp_missing_parameter,
                    make_empty_intermediate_tensors_factory, make_layers)

# Author: Yongjun
import math
import torch.nn.functional as F
from typing import Generator
from vllm.distributed import (differentiable_all_gather,
                              differentiable_identity,
                              differentiable_all_reduce_sum)
from vllm.lora.layers import MergedQKVParallelLinearWithLora


class LlamaMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        bias: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size] * 2,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class LlamaAttention(nn.Module):

    def __init__(
        self,
        config: LlamaConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
        bias: bool = False,
        cache_config: Optional[CacheConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        # MistralConfig has an optional head_dim introduced by Mistral-Nemo
        self.head_dim = getattr(config, "head_dim",
                                self.hidden_size // self.total_num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=hidden_size,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        is_neox_style = True
        if quant_config is not None and quant_config.get_name() == "gguf":
            is_neox_style = False

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=is_neox_style,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
        output, _ = self.o_proj(attn_output)
        return output


class LlamaDecoderLayer(nn.Module):

    def __init__(
        self,
        config: LlamaConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        if rope_scaling is not None and getattr(
                config, "original_max_position_embeddings", None):
            rope_scaling["original_max_position_embeddings"] = (
                config.original_max_position_embeddings)
        max_position_embeddings = getattr(config, "max_position_embeddings",
                                          8192)
        # Support abacusai/Smaug-72B-v0.1 with attention_bias
        # Support internlm/internlm-7b with bias
        attention_bias = getattr(config, "attention_bias", False) or getattr(
            config, "bias", False)
        self.self_attn = LlamaAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=getattr(config, "num_key_value_heads",
                                 config.num_attention_heads),
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            bias=attention_bias,
            cache_config=cache_config,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = LlamaMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            bias=getattr(config, "mlp_bias", False),
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)

        hidden_states = self.self_attn(positions=positions,
                                       hidden_states=hidden_states,
                                       kv_cache=kv_cache,
                                       attn_metadata=attn_metadata)

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": 0,
        "inputs_embeds": 0,
        "intermediate_tensors": 0,
    })
class LlamaModel(nn.Module):

    def __init__(
        self,
        config: LlamaConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        lora_config: Optional[LoRAConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        lora_vocab = (lora_config.lora_extra_vocab_size *
                      (lora_config.max_loras or 1)) if lora_config else 0
        self.vocab_size = config.vocab_size + lora_vocab
        self.org_vocab_size = config.vocab_size
        if get_pp_group().is_first_rank or (config.tie_word_embeddings
                                            and get_pp_group().is_last_rank):
            self.embed_tokens = VocabParallelEmbedding(
                self.vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
                quant_config=quant_config,
            )
        else:
            self.embed_tokens = PPMissingLayer()
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: LlamaDecoderLayer(config=config,
                                             cache_config=cache_config,
                                             quant_config=quant_config,
                                             prefix=prefix),
            prefix=f"{prefix}.layers",
        )
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))
        # Author: Yongjun
        self.make_empty_intermediate_tensors = None

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors],
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            hidden_states, residual = layer(positions, hidden_states,
                                            kv_caches[i - self.start_layer],
                                            attn_metadata, residual)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if ("rotary_emb.cos_cached" in name
                    or "rotary_emb.sin_cached" in name):
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            if scale_name := get_compressed_tensors_cache_scale(name):
                # Loading kv cache scales for compressed-tensors quantization
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                loaded_weight = loaded_weight[0]
                weight_loader(param, loaded_weight)
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)

                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Remapping the name of FP8 kv-scale.
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)

    # If this function is called, it should always initialize KV cache scale
    # factors (or else raise an exception). Thus, handled exceptions should
    # make sure to leave KV cache scale factors in a known good (dummy) state
    def load_kv_cache_scales(self, quantization_param_path: str) -> None:
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        for layer_idx, scaling_factor in kv_cache_scales_loader(
                quantization_param_path, tp_rank, tp_size,
                self.config.num_hidden_layers,
                self.config.__class__.model_type):
            if not isinstance(self.layers[layer_idx], nn.Identity):
                layer_self_attn = self.layers[layer_idx].self_attn

            if is_hip():
                # The scaling factor convention we are assuming is
                # quantized_value * scaling_factor ~= true_value
                # which is consistent with the practice of setting
                # scaling_factor = tensor_amax / FPtype_max
                scaling_factor *= 2
            if hasattr(layer_self_attn, "kv_scale"):
                layer_self_attn.attn._kv_scale = scaling_factor
            else:
                raise RuntimeError("Self attention has no KV cache scaling "
                                   "factor attribute!")


class LlamaForCausalLM(nn.Module, SupportsLoRA, SupportsPP):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"]
    }

    # LoRA specific attributes
    supported_lora_modules = [
        "qkv_proj", "o_proj", "gate_up_proj", "down_proj", "embed_tokens",
        "lm_head"
    ]
    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings"
    }
    embedding_padding_modules = ["lm_head"]

    # BitandBytes specific attributes
    default_bitsandbytes_target_modules = [
        ".gate_proj.",
        ".down_proj.",
        ".up_proj.",
        ".q_proj.",
        ".k_proj.",
        ".v_proj.",
        ".o_proj.",
    ]
    # in TP, these weights are partitioned along the column dimension (dim=-1)
    column_parallel_weights_modules = [".down_proj.", ".o_proj."]
    bitsandbytes_stacked_params_mapping = {
        # shard_name, weight_name, index
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    # Mistral/Llama models can also be loaded with --load-format mistral
    # from consolidated.safetensors checkpoints
    mistral_mapping = {
        "layers": "model.layers",
        "attention": "self_attn",
        "wq": "q_proj",
        "wk": "k_proj",
        "wv": "v_proj",
        "wo": "o_proj",
        "attention_norm": "input_layernorm",
        "feed_forward": "mlp",
        "w1": "gate_proj",
        "w2": "down_proj",
        "w3": "up_proj",
        "ffn_norm": "post_attention_layernorm",
        "tok_embeddings": "model.embed_tokens",
        "output": "lm_head",
        "norm": "model.norm"
    }

    def __init__(
        self,
        config: LlamaConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        lora_config: Optional[LoRAConfig] = None,
    ) -> None:
        super().__init__()

        self.config = config
        self.lora_config = lora_config

        self.model = LlamaModel(config,
                                cache_config,
                                quant_config,
                                lora_config=lora_config,
                                prefix="model")
        if get_pp_group().is_last_rank:
            self.unpadded_vocab_size = config.vocab_size
            if lora_config:
                self.unpadded_vocab_size += lora_config.lora_extra_vocab_size
            self.lm_head = ParallelLMHead(
                self.unpadded_vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
                padding_size=(
                    DEFAULT_VOCAB_PADDING_SIZE
                    # We need bigger padding if using lora for kernel
                    # compatibility
                    if not lora_config else
                    lora_config.lora_vocab_padding_size),
                quant_config=quant_config,
            )
            if config.tie_word_embeddings:
                self.lm_head = self.lm_head.tie_weights(
                    self.model.embed_tokens)

            logit_scale = getattr(config, "logit_scale", 1.0)
            self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                    config.vocab_size,
                                                    logit_scale)
            self.sampler = Sampler()
        else:
            self.lm_head = PPMissingLayer()
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)
        # Author: Yongjun
        self.make_empty_intermediate_tensors = None

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        model_output = self.model(input_ids, positions, kv_caches,
                                  attn_metadata, intermediate_tensors)
        return model_output

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(self, logits: torch.Tensor,
               sampling_metadata: SamplingMetadata) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."]
                           if self.config.tie_word_embeddings else None),
        )
        loader.load_weights(
            self.maybe_remap_mistral(name, loaded_weight)
            for name, loaded_weight in weights)

    def load_kv_cache_scales(self, quantization_param_path: str) -> None:
        self.model.load_kv_cache_scales(quantization_param_path)

    # This function is used to remap the mistral format as
    # used by Mistral and Llama <=2
    def maybe_remap_mistral(
        self,
        name: str,
        loaded_weight: torch.Tensor,
    ) -> Tuple[str, torch.Tensor]:

        def permute(w: torch.Tensor, n_heads: int):
            attn_in = self.config.head_dim * n_heads
            attn_out = self.config.hidden_size

            return w.view(n_heads, attn_in // n_heads // 2, 2,
                          attn_out).transpose(1, 2).reshape(attn_in, attn_out)

        mapping = self.mistral_mapping
        modules = name.split(".")

        # rotary embeds should be sliced
        if "wk" in modules:
            loaded_weight = permute(loaded_weight,
                                    self.config.num_key_value_heads)
        elif "wq" in modules:
            loaded_weight = permute(loaded_weight,
                                    self.config.num_attention_heads)

        for item in modules:
            if item in mapping and mapping[item] not in name:
                name = name.replace(item, mapping[item])

        return name, loaded_weight

    # Author: Yongjun
    def add_lora_train(
        self,
        device: torch.device,
    ) -> None:
        for name, param in self.named_parameters():
            param.requires_grad_(False)

        for name, module in self.named_modules():
            if name.endswith("qkv_proj"):
                torch.manual_seed(0)
                module.lora_a_train_q_proj = torch.nn.Parameter(
                    torch.empty(
                        module.lora_config.max_lora_rank,
                        module.input_size,
                        device=device,
                    ),
                    requires_grad=True,
                )
                torch.nn.init.kaiming_uniform_(module.lora_a_train_q_proj, a=math.sqrt(5))
                module.lora_b_train_q_proj = torch.nn.Parameter(
                    torch.zeros(
                        module.q_proj_shard_size,
                        module.lora_config.max_lora_rank,
                        device=device,
                    ),
                    requires_grad=True,
                )
                module.lora_a_train_k_proj = torch.nn.Parameter(
                    torch.empty(
                        module.lora_config.max_lora_rank,
                        module.input_size,
                        device=device,
                    ),
                    requires_grad=True,
                )
                torch.nn.init.kaiming_uniform_(module.lora_a_train_k_proj, a=math.sqrt(5))
                module.lora_b_train_k_proj = torch.nn.Parameter(
                    torch.zeros(
                        module.kv_proj_shard_size,
                        module.lora_config.max_lora_rank,
                        device=device,
                    ),
                    requires_grad=True,
                )
                module.lora_a_train_v_proj = torch.nn.Parameter(
                    torch.empty(
                        module.lora_config.max_lora_rank,
                        module.input_size,
                        device=device,
                    ),
                    requires_grad=True,
                )
                torch.nn.init.kaiming_uniform_(module.lora_a_train_v_proj, a=math.sqrt(5))
                module.lora_b_train_v_proj = torch.nn.Parameter(
                    torch.zeros(
                        module.kv_proj_shard_size,
                        module.lora_config.max_lora_rank,
                        device=device,
                    ),
                    requires_grad=True,
                )

    def unfused_MergedQKVParallelLinearWithLora(
        self,
        input_: torch.Tensor,
        lora_layer: MergedQKVParallelLinearWithLora,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Matrix multiply.
        query_states = F.linear(
            input_, lora_layer.base_layer.weight[: lora_layer.q_proj_shard_size]
        )
        key_states = F.linear(
            input_,
            lora_layer.base_layer.weight[
                lora_layer.q_proj_shard_size : lora_layer.q_proj_shard_size
                + lora_layer.kv_proj_shard_size
            ],
        )
        value_states = F.linear(
            input_,
            lora_layer.base_layer.weight[
                lora_layer.q_proj_shard_size + lora_layer.kv_proj_shard_size :
            ],
        )

        lora_alpha: int = 32
        scaling = lora_alpha / lora_layer.lora_config.max_lora_rank

        output_dtype = query_states.dtype
        lora_A = lora_layer.lora_a_train_q_proj
        lora_B = lora_layer.lora_b_train_q_proj
        input_ = input_.to(lora_A.dtype)
        after_A = F.linear(input_, lora_A)
        query_states += F.linear(after_A, lora_B) * scaling
        query_states = query_states.to(output_dtype)

        output_dtype = key_states.dtype
        lora_A = lora_layer.lora_a_train_k_proj
        lora_B = lora_layer.lora_b_train_k_proj
        input_ = input_.to(lora_A.dtype)
        after_A = F.linear(input_, lora_A)
        key_states += F.linear(after_A, lora_B) * scaling
        key_states = key_states.to(output_dtype)

        output_dtype = value_states.dtype
        lora_A = lora_layer.lora_a_train_v_proj
        lora_B = lora_layer.lora_b_train_v_proj
        input_ = input_.to(lora_A.dtype)
        after_A = F.linear(input_, lora_A)
        value_states += F.linear(after_A, lora_B) * scaling
        value_states = value_states.to(output_dtype)
        return query_states, key_states, value_states

    def repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
        num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
        """
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

    def unfused_forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Generator[Union[torch.Tensor, IntermediateTensors], None, None]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                # no lora
                hidden_states = self.model.embed_tokens.base_layer(input_ids)
            residual = None
        else:
            raise NotImplementedError

        bsz, q_len = input_ids.size()
        if positions is None:
            positions = torch.arange(0, q_len, dtype=torch.long, device=torch.cuda.current_device())
            positions = positions.unsqueeze(0)

        from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask = attention_mask,
            input_shape = (bsz, q_len),
            inputs_embeds = hidden_states,
            past_key_values_length = 0,
        )

        for i in range(self.model.start_layer, self.model.end_layer):
            layer = self.model.layers[i]

            # Self Attention
            residual = hidden_states
            hidden_states = layer.input_layernorm.forward_native(hidden_states)

            hidden_states = differentiable_identity(hidden_states)
            #q, k, v = torch.utils.checkpoint.checkpoint(
            #    self.unfused_MergedQKVParallelLinearWithLora, hidden_states, layer.self_attn.qkv_proj,
            #    use_reentrant=False
            #)
            q, k, v = self.unfused_MergedQKVParallelLinearWithLora(hidden_states, layer.self_attn.qkv_proj)

            # Naive attention
            # https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
            q = q.view(bsz, q_len, layer.self_attn.num_heads, layer.self_attn.head_dim).transpose(1, 2)
            k = k.view(bsz, q_len, layer.self_attn.num_kv_heads, layer.self_attn.head_dim).transpose(1, 2)
            v = v.view(bsz, q_len, layer.self_attn.num_kv_heads, layer.self_attn.head_dim).transpose(1, 2)

            from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, LlamaRotaryEmbedding
            hf_rotary_emb = LlamaRotaryEmbedding(config=self.model.config).to(torch.cuda.current_device())
            cos, sin = hf_rotary_emb(v, positions)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

            num_key_value_groups = layer.self_attn.num_heads // layer.self_attn.num_kv_heads
            k = self.repeat_kv(k, num_key_value_groups)
            v = self.repeat_kv(v, num_key_value_groups)
            import math
            attn_weights = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(layer.self_attn.head_dim)

            if attention_mask is not None:  # no matter the length, we just slice it
                causal_mask = attention_mask[:, :, :, : k.shape[-2]]
                attn_weights = attn_weights + causal_mask

            # upcast attention to fp32
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
            attn_weights = F.dropout(attn_weights, p=0.0, training=False)
            attn_output = torch.matmul(attn_weights, v)
            if attn_output.size() != (bsz, layer.self_attn.num_heads, q_len, layer.self_attn.head_dim):
                raise ValueError(
                    f"`attn_output` should be of size {(bsz, layer.self_attn.num_heads, q_len, layer.self_attn.head_dim)}, but is"
                    f" {attn_output.size()}"
                )
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, q_len, -1)

            hidden_states = layer.self_attn.o_proj.base_layer.quant_method.apply(
                layer.self_attn.o_proj.base_layer, attn_output
            )
            hidden_states = differentiable_all_reduce_sum(hidden_states)
            hidden_states = residual + hidden_states

            # Fully Connected
            residual = hidden_states
            hidden_states = layer.post_attention_layernorm.forward_native(hidden_states)
            # no lora
            hidden_states = differentiable_identity(hidden_states)
            gate_hidden_states = F.linear(
                hidden_states,
                layer.mlp.gate_up_proj.base_layer.weight[
                    : layer.mlp.gate_up_proj.base_layer.output_partition_sizes[0]
                ],
            )
            up_hidden_states = F.linear(
                hidden_states,
                layer.mlp.gate_up_proj.base_layer.weight[
                    layer.mlp.gate_up_proj.base_layer.output_partition_sizes[0] :
                ],
            )
            hidden_states = F.silu(gate_hidden_states) * up_hidden_states
            hidden_states = layer.mlp.down_proj.base_layer.quant_method.apply(
                layer.mlp.down_proj.base_layer, hidden_states
            )
            hidden_states = differentiable_all_reduce_sum(hidden_states)
            hidden_states = residual + hidden_states

            yield

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })

        hidden_states = self.model.norm.forward_native(hidden_states)
        logits = F.linear(hidden_states, self.lm_head.weight).float()
        if self.lm_head.tp_size > 1:
            logits = differentiable_all_gather(logits)
        logits = logits[:, :, : self.model.config.vocab_size]

        yield logits

    def fused_MergedQKVParallelLinearWithLora(
        self,
        f_input: torch.Tensor,
        i_input: torch.Tensor,
        lora_layer: MergedQKVParallelLinearWithLora,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # End-to-end forward speeds up by 1.2%
        f_bsz, f_q_len, _ = f_input.size()

        # Matrix multiply.
        fi_input = torch.cat((f_input, i_input.unsqueeze(dim=0)), dim=1)
        fi_qkv = lora_layer.base_layer.quant_method.apply(lora_layer.base_layer, fi_input)
        f_q, f_k, f_v = fi_qkv[:, :f_q_len, :].split([lora_layer.q_proj_shard_size,
            lora_layer.kv_proj_shard_size, lora_layer.kv_proj_shard_size], dim=-1)

        lora_alpha: int = 32
        scaling = lora_alpha / lora_layer.lora_config.max_lora_rank

        output_dtype = f_q.dtype
        lora_A = lora_layer.lora_a_train_q_proj
        lora_B = lora_layer.lora_b_train_q_proj
        after_A = F.linear(f_input.to(lora_A.dtype), lora_A)
        f_q = f_q + F.linear(after_A, lora_B) * scaling
        f_q = f_q.to(output_dtype)

        output_dtype = f_k.dtype
        lora_A = lora_layer.lora_a_train_k_proj
        lora_B = lora_layer.lora_b_train_k_proj
        after_A = F.linear(f_input.to(lora_A.dtype), lora_A)
        f_k = f_k + F.linear(after_A, lora_B) * scaling
        f_k = f_k.to(output_dtype)

        output_dtype = f_v.dtype
        lora_A = lora_layer.lora_a_train_v_proj
        lora_B = lora_layer.lora_b_train_v_proj
        after_A = F.linear(f_input.to(lora_A.dtype), lora_A)
        f_v = f_v + F.linear(after_A, lora_B) * scaling
        f_v = f_v.to(output_dtype)

        i_qkv = fi_qkv[0, f_q_len:, :].detach()
        lora_layer.punica_wrapper.add_lora_packed_nslice(
            i_qkv, i_input,
            lora_layer.lora_a_stacked, lora_layer.lora_b_stacked, 1.0,
            lora_layer.output_slices
        )
        i_q, i_k, i_v = i_qkv.split([lora_layer.q_proj_shard_size,
            lora_layer.kv_proj_shard_size, lora_layer.kv_proj_shard_size], dim=-1)

        return f_q, f_k, f_v, i_q, i_k, i_v

    def fused_forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        positions: torch.Tensor,
        i_input_ids: Optional[torch.Tensor] = None,
        i_positions: Optional[torch.Tensor] = None,
        kv_caches: Optional[List[torch.Tensor]] = None,
        attn_metadata: Optional[AttentionMetadata] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        # no lora
        hidden_states = self.model.embed_tokens.base_layer(input_ids)
        i_hidden_states = self.model.embed_tokens.base_layer(i_input_ids)

        residual = None
        i_residual = None

        bsz, q_len = input_ids.size()
        if positions is None:
            positions = torch.arange(0, q_len, dtype=torch.long, device=torch.cuda.current_device())
            positions = positions.unsqueeze(0)

        from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask_for_sdpa
        attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
            attention_mask = attention_mask,
            input_shape = (bsz, q_len),
            inputs_embeds = hidden_states,
            past_key_values_length = 0,
        )

        for i in range(self.model.start_layer, self.model.end_layer):
            layer = self.model.layers[i]

            # Self Attention
            residual = hidden_states
            hidden_states = layer.input_layernorm.forward_native(hidden_states)
            if i_residual is None:
                i_residual = i_hidden_states
                i_hidden_states = layer.input_layernorm(i_hidden_states)
            else:
                i_hidden_states, i_residual = layer.input_layernorm(
                    i_hidden_states, i_residual)

            hidden_states = differentiable_identity(hidden_states)
            q, k, v, i_q, i_k, i_v = self.fused_MergedQKVParallelLinearWithLora(
                hidden_states, i_hidden_states, layer.self_attn.qkv_proj
            )
            q, k = layer.self_attn.rotary_emb(positions, q, k)

            # PyTorch LlamaSdpaAttention
            # https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
            q = q.view(bsz, q_len, layer.self_attn.num_heads, layer.self_attn.head_dim).transpose(1, 2)
            k = k.view(bsz, q_len, layer.self_attn.num_kv_heads, layer.self_attn.head_dim).transpose(1, 2)
            v = v.view(bsz, q_len, layer.self_attn.num_kv_heads, layer.self_attn.head_dim).transpose(1, 2)

            num_key_value_groups = layer.self_attn.num_heads // layer.self_attn.num_kv_heads
            k = self.repeat_kv(k, num_key_value_groups)
            v = self.repeat_kv(v, num_key_value_groups)

            causal_mask = attention_mask
            if attention_mask is not None:  # no matter the length, we just slice it
                causal_mask = attention_mask[:, :, :, : k.shape[-2]]

            if q.device.type == "cuda" and causal_mask is not None:
                q = q.contiguous()
                k = k.contiguous()
                v = v.contiguous()

            torch.backends.cuda.enable_flash_sdp(True)
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=causal_mask,
                dropout_p=0.0,
                is_causal=False,
            )

            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.view(bsz, q_len, -1)

            hidden_states = layer.self_attn.o_proj.base_layer.quant_method.apply(
                layer.self_attn.o_proj.base_layer, attn_output
            )
            hidden_states = differentiable_all_reduce_sum(hidden_states)
            hidden_states = residual + hidden_states

            # vLLM LlamaAttention
            # https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/llama.py
            i_q, i_k = layer.self_attn.rotary_emb(i_positions, i_q, i_k)
            attn_output = layer.self_attn.attn(i_q, i_k, i_v,
                kv_caches[i - self.model.start_layer], attn_metadata)
            i_hidden_states, _ = layer.self_attn.o_proj(attn_output)

            # Fully Connected
            residual = hidden_states
            hidden_states = layer.post_attention_layernorm.forward_native(hidden_states)
            i_hidden_states, i_residual = layer.post_attention_layernorm(
                i_hidden_states, i_residual)

            # no lora
            hidden_states = differentiable_identity(hidden_states)

            # End-to-end forward speeds up by 10.2%
            fi_hidden_states = torch.cat((hidden_states, i_hidden_states.unsqueeze(dim=0)), dim=1)
            gate_up = layer.mlp.gate_up_proj.base_layer.quant_method.apply(
                layer.mlp.gate_up_proj.base_layer, fi_hidden_states
            )
            fi_hidden_states = layer.mlp.act_fn.forward_native(gate_up)
            fi_hidden_states = layer.mlp.down_proj.base_layer.quant_method.apply(
                layer.mlp.down_proj.base_layer, fi_hidden_states
            )
            i_hidden_states = fi_hidden_states[0, q_len:, :].detach()
            hidden_states = fi_hidden_states[:, :q_len, :]

            hidden_states = differentiable_all_reduce_sum(hidden_states)
            hidden_states = residual + hidden_states

        hidden_states = self.model.norm.forward_native(hidden_states)
        logits = F.linear(hidden_states, self.lm_head.weight).float()
        if self.lm_head.tp_size > 1:
            logits = differentiable_all_gather(logits)
        logits = logits[:, :, : self.model.config.vocab_size]

        i_hidden_states, _ = self.model.norm(i_hidden_states, i_residual)
        return logits, i_hidden_states
