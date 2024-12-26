import os
import gc
import argparse
import itertools
from typing import List, Dict, Tuple, Union

import torch
from tqdm import tqdm

from benchmarks.chunked_peft.data_utils import (
    load_data,
)

from benchmarks.chunked_peft.chunk_utils import (
    printer,
    fix_randomness_determinism,
    create_chunk_model,
    get_dtype,
)

from benchmarks.chunked_peft.timer_utils import (
    get_stat_str,
)

def whole_peft_forward(model: torch.nn.Module, 
                       inputs: Union[List[int], Dict],
                      ) -> Tuple[List[torch.nn.Module], torch.Tensor]:
    inputs.to(torch.cuda.current_device())
    outputs = model(**inputs, use_cache=False)
    return outputs.loss, None

def whole_peft_backward(model: torch.nn.Module,
                        loss: torch.nn.Module,
                        check_grad: bool = False) -> None:
    loss.backward()

def run_whole_peft_step(model: torch.nn.Module,
                        optimizer: torch.optim.Optimizer,
                        step: int,
                        inputs: Union[List[int], Dict],
                        do_backward: bool,
                        gradient_accumulation_steps: int,
                        local_rank: int,
                        check_grad: bool = False,
                        forward_timings: List[float] = [],
                        total_timings: List[float] = [],
                        ) -> None:
    torch.cuda.set_device(local_rank)
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_forward_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    
    loss, _ = whole_peft_forward(model=model, inputs=inputs)
    
    end_forward_event.record()
    torch.cuda.synchronize()
    forward_timings.append(start_event.elapsed_time(end_forward_event) / 1000.0)
    
    if do_backward:
        whole_peft_backward(model=model, loss=loss, check_grad=check_grad)
        if step % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
    
    end_event.record()
    torch.cuda.synchronize()
    total_timings.append(start_event.elapsed_time(end_event) / 1000.0)


def run_whole_peft_holistic(model: torch.nn.Module, 
                            optimizer: torch.optim.Optimizer,
                            dataloader_iter: itertools.cycle, 
                            num_run_steps: int,
                            do_backward: bool,
                            gradient_accumulation_steps: int,
                            seq_len: int,
                            chunk_size: int,
                            local_rank: int,
                            check_grad: bool = False,
                            forward_timings: List[float] = [],
                            total_timings: List[float] = [],
                            ) -> None:
    valid_step: int = 0
    for step, inputs in tqdm(dataloader_iter):
        run_whole_peft_step(model, optimizer, valid_step, inputs,
                            do_backward=do_backward,
                            gradient_accumulation_steps=gradient_accumulation_steps,
                            local_rank=local_rank,
                            check_grad=check_grad,
                            forward_timings=forward_timings,
                            total_timings=total_timings,)
        # check break cond
        valid_step += 1
        if valid_step >= num_run_steps:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", "-m", type=str, default="meta-llama/Meta-Llama-3.1-8B", help="model name or path")
    parser.add_argument("--dataset_name", type=str, default="yahma/alpaca-cleaned", help="dataset name or path")
    parser.add_argument("--warmup", type=int, default=0,  help="Number of warmup peft iterations for benchmarking")
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
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32", "fp64"])
    parser.add_argument("--attn_impl", type=str, default="flash_attention_2", choices=["eager", "flash_attention_2", "math", "mem_efficient"], help="torch attention implementation")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--res_folder", type=str, default=".", help="path to store benchmarking results")
    parser.add_argument("--print_out", type=str, default="y", help="y: print info; n: no show")
    args = parser.parse_args()

    gc.collect()

    # fix random seed
    fix_randomness_determinism(seed=args.seed, deterministic=False)

    # open printer
    printer.open() if args.print_out == "y" else printer.close()

    # timer records
    forward_timings: List[float] = []
    total_timings:   List[float] = []

    # create dataloader
    dataloader_iter = load_data(model_name=args.model_name, 
                                dataset_name=args.dataset_name, 
                                cache_dir=args.cache_dir, 
                                seq_len=args.seq_len, 
                                batch_size=1, 
                                pin_memory=bool(args.pin_memory),
                                shuffle=False)
    
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
                            seq_len=args.seq_len,
                            chunk_size=args.chunk_size,
                            local_rank=args.local_rank,
                            check_grad=False,
                            forward_timings=forward_timings,
                            total_timings=total_timings)
    
    # get results
    assert args.warmup < args.trials, "warm-up steps should be smaller than total steps..."
    log_str: str = get_stat_str(model_name=args.model_name,
                                cache_dir=args.cache_dir,
                                batch_size=args.batch_size,
                                warmup_steps=args.warmup,
                                forward_timings=forward_timings,
                                total_timings=total_timings,)
    fname: str = f"whole_{args.dtype}_seqlen{args.seq_len}_trials{args.trials}.txt"
    with open(args.res_folder + "/" + fname, "a+") as fp:
        seed_str: str = f"[seed: {args.seed}]\n"
        fp.write(seed_str + log_str + '\n')