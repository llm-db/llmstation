import transformers.loss.loss_utils
import transformers.models.qwen2.modeling_qwen2
import transformers.models.llama.modeling_llama

from benchmarks.chunked_peft.monkey_patching.loss_utils import ForCausalLMLoss
from benchmarks.chunked_peft.monkey_patching.qwen2_correctness import (
    Qwen2RMSNorm,
    Qwen2Attention,
    Qwen2ChunkFlashAttention2,
    Qwen2ChunkSdpaAttention,
    Qwen2Model,
)
from benchmarks.chunked_peft.monkey_patching.llama_performance import (
    LlamaChunkFlashAttention2,
    LlamaChunkSdpaAttention,
    LlamaModel,
)

def open_correctness_mode() -> None:
    print(f"using monkey patching to overwrite transformers Qwen2 code for correctness verification of chunked PEFT...")
    transformers.loss.loss_utils.LOSS_MAPPING["ForCausalLM"] = ForCausalLMLoss
    # transformers.loss.loss_utils.ForCausalLMLoss = ForCausalLMLoss
    transformers.models.qwen2.modeling_qwen2.Qwen2RMSNorm = Qwen2RMSNorm
    transformers.models.qwen2.modeling_qwen2.Qwen2Attention = Qwen2Attention
    transformers.models.qwen2.modeling_qwen2.Qwen2Model = Qwen2Model
    transformers.models.qwen2.modeling_qwen2.QWEN2_ATTENTION_CLASSES["eager"] = Qwen2Attention
    transformers.models.qwen2.modeling_qwen2.QWEN2_ATTENTION_CLASSES["flash_attention_2"] = Qwen2ChunkFlashAttention2
    transformers.models.qwen2.modeling_qwen2.QWEN2_ATTENTION_CLASSES["sdpa"] = Qwen2ChunkSdpaAttention

def open_performance_mode() -> None:
    print(f"using monkey patching to overwrite transformers Llama code for performance measurement of chunked PEFT...")
    transformers.models.llama.modeling_llama.LlamaModel = LlamaModel
    transformers.models.llama.modeling_llama.LLAMA_ATTENTION_CLASSES["flash_attention_2"] = LlamaChunkFlashAttention2
    transformers.models.llama.modeling_llama.LLAMA_ATTENTION_CLASSES["sdpa"] = LlamaChunkSdpaAttention
