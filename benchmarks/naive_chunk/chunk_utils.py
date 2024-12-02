import os
import random
from collections import OrderedDict
from typing import List, Dict, Tuple, Union, Optional

import torch
import numpy as np
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
)
from peft import LoraConfig, get_peft_model

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

def get_hf_model(
    model_name: str,
    pin_memory: Union[str, bool],
    quant_bits: int,
    quant_group_size: int,
    cache_dir: str,
    dtype: torch.dtype = torch.bfloat16,
    attn_impl: str = "eager",
    lora_r: int = 64,
    lora_alpha: int = 32,
    lora_dropout: float = 0.0,
    target_modules: List[str] = ["q_proj", "v_proj"],
) -> torch.nn.Module:
    config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
    config._attn_implementation = attn_impl
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
                       attn_impl: str = "eager",
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
        model: torch.nn.Module = get_hf_model(
            model_name,
            pin_memory,
            quant_bits,
            quant_group_size,
            cache_dir,
            dtype=dtype,
            attn_impl=attn_impl,
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