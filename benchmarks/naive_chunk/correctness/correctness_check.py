import os
import gc
import argparse
import itertools
from collections import OrderedDict
from typing import List, Tuple, Optional

import torch
from tqdm import tqdm

from benchmarks.naive_chunk.data_utils import (
    load_data,
)

from benchmarks.naive_chunk.chunk_utils import (
    printer,
    fix_determinism,
    copy_input,
    create_chunk_model,
    get_dtype,
)

from benchmarks.naive_chunk.correctness.whole_peft import (
    run_whole_peft_step
)

from benchmarks.naive_chunk.correctness.chunked_peft import (
    run_chunked_peft_step
)

def check_results(res_tuple_1, res_tuple_2, tolerance: float = 1e-5, 
                  check_grad: bool = False, grad_tolerance=1e-6):
    printer.print("checking generated losses and logits...")
    loss_1,   loss_2   = res_tuple_1[0], res_tuple_2[0]
    logits_1, logits_2 = res_tuple_1[1], res_tuple_2[1]
    printer.print(f"losses 1 shape: {loss_1.shape}"), printer.print(f"logits 1 shape: {logits_1.shape}")
    printer.print(f"losses 2 shape: {loss_2.shape}"), printer.print(f"logits 2 shape: {logits_2.shape}")
    printer.print(f"losses 1: {loss_1}"), printer.print(f"logits 1: {logits_1}")
    printer.print(f"losses 2: {loss_2}"), printer.print(f"logits 2: {logits_2}")
    assert torch.allclose(loss_1.cpu(), loss_2.cpu(), atol=tolerance)
    assert torch.allclose(logits_1.cpu(), logits_2.cpu(), atol=tolerance, rtol=tolerance)
    if check_grad:
        grad_dict_1, grad_dict_2 = res_tuple_1[2], res_tuple_2[2]
        printer.print(f"grad_dict_1: {grad_dict_1}"), printer.print(f"grad_dict_2: {grad_dict_2}")
        for name, _ in reversed(list(grad_dict_1.items())):
            assert torch.allclose(grad_dict_1[name].cpu(), grad_dict_2[name].cpu(), 
                                    atol=grad_tolerance, rtol=grad_tolerance), \
                    f"[{name}] model 1 grad: {grad_dict_1[name][:-1,:]}; model 2 grad: {grad_dict_2[name][:-1,:]}"
            printer.print(f"param: {name} pass allclose check...")

def free_results(result_tuple):
    for cur_item in result_tuple:
        del cur_item
    torch.cuda.empty_cache()

def move_results(result_tuple):
    for cur_item in result_tuple:
        if isinstance(cur_item, torch.Tensor):
            cur_item.to(device="cpu")
        else:
            for key, value in cur_item.items():
                value.to(device="cpu")
    torch.cuda.empty_cache()
    return result_tuple

def run_and_check_results(models: Tuple[torch.nn.Module, torch.nn.Module], 
                          optimizers: Tuple[torch.optim.Optimizer,torch.optim.Optimizer],
                          dataloader_iter: itertools.cycle, 
                          num_run_steps: int,
                          do_backward: bool,
                          gradient_accumulation_steps: int,
                          seq_len: int,
                          chunk_size: int,
                          local_ranks: Tuple[int, int],
                          tolerances: Tuple[float, float],
                          check_grad: bool = False,
                          backward_whole: bool = False,
                          backward_order: bool = True,
                          ) -> List[Tuple[torch.tensor, torch.tensor, Optional[OrderedDict]]]:
    results: List[Tuple[torch.tensor, torch.tensor, Optional[OrderedDict]]] = []
    valid_step: int = 0
    num_chunks: int = seq_len // chunk_size
    len_thresh: int = chunk_size * (num_chunks-1)
    for step, inputs in tqdm(dataloader_iter):
        # skip not-long-enough samples to ensure the full chunk number
        if inputs.input_ids.shape[1] <= len_thresh:
            continue
        # first: whole
        whole_inputs = copy_input(inputs, local_rank=local_ranks[0])
        whole_result = run_whole_peft_step(models[0], optimizers[0], valid_step, whole_inputs,
                                           do_backward=do_backward,
                                           gradient_accumulation_steps=gradient_accumulation_steps,
                                           local_rank=local_ranks[0],
                                           check_grad=check_grad)
        # second: chunk
        chunk_inputs = copy_input(inputs, local_rank=local_ranks[1])
        chunk_result = run_chunked_peft_step(models[1], optimizers[1], valid_step, chunk_inputs,
                                             do_backward=do_backward,
                                             gradient_accumulation_steps=gradient_accumulation_steps,
                                             chunk_size=chunk_size,
                                             local_rank=local_ranks[1],
                                             check_grad=check_grad,
                                             backward_whole=backward_whole,
                                             backward_order=backward_order)
        # check result
        check_results(whole_result, chunk_result, tolerance=tolerances[0], 
                      check_grad=check_grad, grad_tolerance=tolerances[1])
        # free result
        free_results(whole_result), free_results(chunk_result)
        gc.collect()
        # check break cond
        valid_step += 1
        if valid_step >= num_run_steps:
            break
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", "-m", type=str, default="meta-llama/Meta-Llama-3.1-8B", help="model name or path")
    parser.add_argument("--dataset_name", type=str, default="yahma/alpaca-cleaned", help="dataset name or path")
    parser.add_argument("--trials", type=int, default=5,  help="Number of peft iterations")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=256,  help="sequence length")
    parser.add_argument("--chunk_size", type=int, default=128, help="token chunk size")
    parser.add_argument("--local_ranks", type=int, nargs='+', help="local ranks for comparison")
    parser.add_argument("--pin_memory", type=int, default=0, help="whether to pinned CPU memory for ZeRO offloading")
    parser.add_argument("--quant_bits", type=int, default=16, help="model weight quantization bits; either 4 or 8")
    parser.add_argument("--quant_group_size", type=int, default=64, help="model weight quantization group size")
    parser.add_argument("--cache_dir", type=str, default=".", help="cache dir for model name")
    parser.add_argument("--dtype", type=str, default="fp64", choices=["fp16", "bf16", "fp32", "fp64"])
    parser.add_argument("--tolerances", type=float, nargs='+', help="2-element tuple: first is loss/logits tolerance; second is grads tolerance")
    parser.add_argument("--print_out", type=str, default="y", help="y: print info; n: no show")
    args = parser.parse_args()

    gc.collect()

    # fix random seed
    fix_determinism()

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
    models = create_chunk_model(model_name=args.model_name,
                                pin_memory=bool(args.pin_memory),
                                quant_bits=args.quant_bits,
                                quant_group_size=args.quant_group_size,
                                cache_dir=args.cache_dir,
                                local_ranks=args.local_ranks,
                                dtype=get_dtype(args.dtype),
                                attn_impl="eager",)
                                # attn_impl="flash_attention_2",)
    # create optimizer
    optimizers = (torch.optim.AdamW(models[0].parameters(), lr=3e-4), \
                  torch.optim.AdamW(models[1].parameters(), lr=3e-4),)
    # run chunked peft
    run_and_check_results(models=models, 
                          optimizers=optimizers,
                          dataloader_iter=dataloader_iter,
                          num_run_steps=args.trials,
                          do_backward=True,
                          gradient_accumulation_steps=args.gradient_accumulation_steps,
                          seq_len=args.seq_len,
                          chunk_size=args.chunk_size,
                          tolerances=tuple(args.tolerances),
                          local_ranks=args.local_ranks,
                          check_grad=True,
                          backward_whole=False,
                          backward_order=True)