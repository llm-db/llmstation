import os
import gc
import argparse
import itertools
from collections import OrderedDict
from typing import List, Dict, Tuple, Union, Optional

import torch
from tqdm import tqdm

from benchmarks.chunked_peft.data_utils import (
    load_data,
)

from benchmarks.chunked_peft.chunk_utils import (
    printer,
    fix_randomness_determinism,
    get_learnable_param_grad,
    create_chunk_model,
    get_dtype,
)


def whole_peft_forward(model: torch.nn.Module, 
                       inputs: Union[List[int], Dict],
                      ) -> Tuple[List[torch.nn.Module], torch.Tensor]:
    inputs.to(torch.cuda.current_device())
    outputs = model(**inputs, use_cache=False)
    return outputs.loss, outputs.logits

def whole_peft_backward(model: torch.nn.Module,
                        loss: torch.nn.Module,
                        check_grad: bool = False) -> None:
    printer.print(f"bwd pass in the whole")
    loss.backward()
    printer.print(f"grad of last layer v.proj: {model.base_model.        \
                                                model.model.layers[-1].  \
                                                self_attn.v_proj.lora_B. \
                                                default.weight.grad}")

def run_whole_peft_step(model: torch.nn.Module,
                        optimizer: torch.optim.Optimizer,
                        step: int,
                        inputs: Union[List[int], Dict],
                        do_backward: bool,
                        gradient_accumulation_steps: int,
                        local_rank: int,
                        check_grad: bool = False,
                        ) -> Tuple[torch.tensor, torch.tensor, Optional[OrderedDict]]:
    torch.cuda.set_device(local_rank)
    loss, logits = whole_peft_forward(model=model, inputs=inputs)
    torch.cuda.synchronize()
    result = (loss, logits, )
    if do_backward:
        whole_peft_backward(model=model, loss=loss, check_grad=check_grad)
        if check_grad:
            result += (get_learnable_param_grad(model), )
        if step % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
    torch.cuda.synchronize()
    return result

def run_whole_peft_holistic(model: torch.nn.Module, 
                            optimizer: torch.optim.Optimizer,
                            dataloader_iter: itertools.cycle, 
                            num_run_steps: int,
                            do_backward: bool,
                            gradient_accumulation_steps: int,
                            local_rank: int,
                            check_grad: bool = False,
                            ) -> None:
    for step, inputs in tqdm(dataloader_iter):
        run_whole_peft_step(model, optimizer, step, inputs,
                            do_backward=do_backward,
                            gradient_accumulation_steps=gradient_accumulation_steps,
                            local_rank=local_rank,
                            check_grad=check_grad,)
        if step >= num_run_steps:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", "-m", type=str, default="meta-llama/Meta-Llama-3.1-8B", help="model name or path")
    parser.add_argument("--dataset_name", type=str, default="yahma/alpaca-cleaned", help="dataset name or path")
    parser.add_argument("--trials", type=int, default=5,  help="Number of peft iterations")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=256,  help="sequence length")
    parser.add_argument("--chunk_size", type=int, default=128, help="token chunk size")
    parser.add_argument("--local_rank", type=int, default=int(os.getenv("LOCAL_RANK", "0")), help="local rank for distributed inference")
    parser.add_argument("--pin_memory", type=int, default=0, help="whether to pinned CPU memory for ZeRO offloading")
    parser.add_argument("--quant_bits", type=int, default=16, help="model weight quantization bits; either 4 or 8")
    parser.add_argument("--quant_group_size", type=int, default=64, help="model weight quantization group size")
    parser.add_argument("--cache_dir", type=str, default=".", help="cache dir for model name")
    parser.add_argument("--dtype", type=str, default="fp64", choices=["fp16", "bf16", "fp32", "fp64"])
    parser.add_argument("--attn_impl", type=str, default="flash_attention_2", choices=["eager", "flash_attention_2", "sdpa"], help="attention implementation")
    parser.add_argument("--print_out", type=str, default="y", help="y: print info; n: no show")
    args = parser.parse_args()

    gc.collect()

    # fix random seed
    fix_randomness_determinism()

    # open printer
    printer.open() if args.print_out == "y" else printer.close()

    # create dataloader
    dataloader_iter = load_data(model_name=args.model_name, 
                                dataset_name=args.dataset_name, 
                                cache_dir=args.cache_dir, 
                                seq_len=args.seq_len, 
                                batch_size=1, 
                                pin_memory=bool(args.pin_memory))
    # create model
    model = create_chunk_model(model_name=args.model_name,
                               pin_memory=bool(args.pin_memory),
                               quant_bits=args.quant_bits,
                               quant_group_size=args.quant_group_size,
                               cache_dir=args.cache_dir,
                               local_rank=args.local_rank,
                               dtype=get_dtype(args.dtype),
                               attn_impl=args.attn_impl,)
    # create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    # run whole peft
    run_whole_peft_holistic(model=model, 
                            optimizer=optimizer,
                            dataloader_iter=dataloader_iter,
                            num_run_steps=args.trials,
                            do_backward=True,
                            gradient_accumulation_steps=args.gradient_accumulation_steps,
                            local_rank=args.local_rank,
                            check_grad=True,)