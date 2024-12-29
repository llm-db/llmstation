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
    ChunkCache,
    INVALID_TARGET,
    fix_randomness_determinism,
    fixed_cross_entropy,
    create_chunk_model,
    get_dtype,
)

from benchmarks.chunked_peft.timer_utils import (
    get_stat_str,
)

from benchmarks.chunked_peft.monkey_patching import open_performance_mode

def chunked_peft_forward(
    model: torch.nn.Module, 
    inputs: Union[List[int], Dict],
    chunk_size: int,
    start_event: torch.cuda.Event = None,
) -> Tuple[List[torch.nn.Module], torch.Tensor]:
    assert chunk_size >= 1, "chunk size should be a postive integer"
    num_total_tokens: int = inputs.input_ids.shape[1]
    num_processed_tokens: int = 0
    past_key_values = ChunkCache(attn_impl=model.config._attn_implementation).to(device=torch.cuda.current_device())
    num_valid_tokens: torch.Tensor = (inputs.labels != INVALID_TARGET).sum(dim=-1, keepdim=True)\
                                                                      .to(device=torch.cuda.current_device()) - 1
    losses: List[torch.nn.Module] = []
    
    elapsed_times: List[float] = [ ]
    if start_event is not None:
        init_event = torch.cuda.Event(enable_timing=True)
        init_event.record()
        torch.cuda.synchronize()
        prev_elapsed_time: float = start_event.elapsed_time(init_event) / 1000
    
    while num_processed_tokens < num_total_tokens:
        if start_event is not None:
            cur_event = torch.cuda.Event(enable_timing=True)
        
        num_cur_step: int = min(chunk_size, num_total_tokens-num_processed_tokens)
        num_will_complete_tokens = num_processed_tokens + num_cur_step
        cache_position: torch.Tensor = torch.arange(start=num_processed_tokens, 
                                                    end=num_will_complete_tokens,
                                                    device=torch.cuda.current_device())
        attention_mask: torch.Tensor = inputs.attention_mask[:, :num_will_complete_tokens]
        cur_input_ids: torch.Tensor  = inputs.input_ids[:, num_processed_tokens: num_will_complete_tokens]
        cur_inputs: Dict = {"input_ids": cur_input_ids.to(device=torch.cuda.current_device()),
                            "attention_mask": attention_mask.to(device=torch.cuda.current_device()),
                            "cache_position": cache_position,
                            "past_key_values": past_key_values,
                            "use_cache": True,
                            }
        
        
        outputs = model(**cur_inputs)

        # NOTE: detach prev tensor from the computation graph of new chunks
        past_key_values: ChunkCache = outputs.past_key_values
        past_key_values.detach_kv_cache()
        
        cur_logits: torch.Tensor = outputs.logits
        
        # (batch size, len, class) -> (batch size, class, len) -> (batch size, len) -> (batch size, 1)
        if num_will_complete_tokens < num_total_tokens:
            shift_logits:  torch.Tensor = cur_logits[..., :, :].contiguous()
            shift_targets: torch.Tensor = inputs.labels[:, num_processed_tokens+1: num_will_complete_tokens+1] \
                                                .to(device=torch.cuda.current_device())
        else: # last chunk, ignore the last pos of logits
            shift_logits:  torch.Tensor = cur_logits[..., :-1, :].contiguous()
            shift_targets: torch.Tensor = inputs.labels[:, num_processed_tokens+1: num_will_complete_tokens] \
                                                .to(device=torch.cuda.current_device())
        
        # NOTE: here we only consider batch_size = 1, as for FT-serving purpose
        #       a smaller batch (i.e., 1) rather than smaller chunks should be the first to consider
        shift_targets = shift_targets.view(-1)
        shift_logits  = shift_logits.view(-1, shift_logits.shape[-1])
        shift_targets = shift_targets.to(shift_logits.device)
        cur_loss = fixed_cross_entropy(shift_logits, shift_targets, 1, INVALID_TARGET)
        
        cur_loss /= num_valid_tokens.item() # sum of avg-token-loss in current chunk
        losses.append(cur_loss)
        num_processed_tokens += num_cur_step

        if start_event is not None:
            cur_event.record()
            torch.cuda.synchronize()
            cur_elapsed_time: float = start_event.elapsed_time(cur_event) / 1000.0
            elapsed_times.append(cur_elapsed_time - prev_elapsed_time)
            prev_elapsed_time = cur_elapsed_time

    return losses, elapsed_times


def chunked_peft_backward(
    losses: List[torch.nn.Module],
    start_event: torch.cuda.Event = None,
    end_forward_event: torch.cuda.Event = None,
) -> List[float]:
    elapsed_times: List[float] = [ ]
    if start_event is not None:
        prev_elapsed_time: float = start_event.elapsed_time(end_forward_event) / 1000
    
    for loss in losses[::-1]:
        if start_event is not None:
            cur_event = torch.cuda.Event(enable_timing=True)

        # NOTE: retain_graph = True is a bit faster w/ higher peak gpu mem
        #       retain_graph = False is slower w/ lower peak gpu mem
        loss.backward(retain_graph=False)

        if start_event is not None:
            cur_event.record()
            torch.cuda.synchronize()
            cur_elapsed_time: float = start_event.elapsed_time(cur_event) / 1000.0
            elapsed_times.append(cur_elapsed_time - prev_elapsed_time)
            prev_elapsed_time = cur_elapsed_time

    return elapsed_times


def run_chunked_peft_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    inputs: Union[List[int], Dict],
    gradient_accumulation_steps: int,
    chunk_size: int,
    local_rank: int,
    forward_timings: List[float] = [],
    total_timings: List[float] = [],
    show_chunk_time: str = "n",
    fwd_chunk_timings: List[List[float]] = [],
    bwd_chunk_timings: List[List[float]] = [],
) -> None:
    torch.cuda.set_device(local_rank)

    start_event = torch.cuda.Event(enable_timing=True)
    end_forward_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()

    losses, per_chunk_fwd_timings = chunked_peft_forward(
        model=model, inputs=inputs, chunk_size=chunk_size, 
        start_event=start_event if show_chunk_time == "y" else None
    )

    end_forward_event.record()
    torch.cuda.synchronize()
    forward_timings.append(start_event.elapsed_time(end_forward_event) / 1000.0)

    if show_chunk_time == "n":
        chunked_peft_backward(losses=losses)
    else:
        per_chunk_bwd_timings: List[float] = chunked_peft_backward(
            losses=losses, start_event=start_event, end_forward_event=end_forward_event
        )
          
    if step % gradient_accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

    end_event.record()
    torch.cuda.synchronize()
    total_timings.append(start_event.elapsed_time(end_event) / 1000.0)

    if show_chunk_time == "y" and per_chunk_fwd_timings is not None:
        num_chunks: int = len(per_chunk_fwd_timings)
        for idx, elapsed_time in enumerate(per_chunk_fwd_timings):
            print(f"fwd time of {idx}-th chunk: {elapsed_time}")

    if show_chunk_time == "y" and per_chunk_bwd_timings is not None:
        num_chunks: int = len(per_chunk_bwd_timings)
        for idx, elapsed_time in enumerate(per_chunk_bwd_timings):
            print(f"bwd time of {num_chunks-idx-1}-th chunk: {elapsed_time}")
    
    if show_chunk_time == "y":
        fwd_chunk_timings.append(per_chunk_fwd_timings)
        bwd_chunk_timings.append(per_chunk_bwd_timings)


def run_chunked_peft_holistic(
    model: torch.nn.Module, 
    optimizer: torch.optim.Optimizer,
    dataloader_iter: itertools.cycle, 
    num_run_steps: int,
    gradient_accumulation_steps: int,
    chunk_size: int,
    local_rank: int,
    forward_timings: List[float] = [],
    total_timings: List[float] = [],
    show_chunk_time: str = "n",
    fwd_chunk_timings: List[List[float]] = [],
    bwd_chunk_timings: List[List[float]] = [],
) -> None:
    valid_step: int = 0
    for step, inputs in tqdm(dataloader_iter):
        run_chunked_peft_step(
            model, optimizer, valid_step, inputs,
            gradient_accumulation_steps=gradient_accumulation_steps,
            chunk_size=chunk_size,
            local_rank=local_rank,
            forward_timings=forward_timings,
            total_timings=total_timings,
            show_chunk_time=show_chunk_time,
            fwd_chunk_timings=fwd_chunk_timings,
            bwd_chunk_timings=bwd_chunk_timings,
        )
        # check break cond
        valid_step += 1
        if valid_step >= num_run_steps:
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", "-m", type=str, default="meta-llama/Meta-Llama-3.1-8B", help="model name or path")
    parser.add_argument("--dataset_name", type=str, default="yahma/alpaca-cleaned", help="dataset name or path")
    parser.add_argument("--warmup", type=int, default=10,  help="Number of warmup peft iterations for benchmarking")
    parser.add_argument("--trials", type=int, default=30,  help="Number of peft iterations")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=256,  help="sequence length")
    parser.add_argument("--chunk_size", type=int, default=128, help="token chunk size")
    parser.add_argument("--local_rank", type=int, default=int(os.getenv("LOCAL_RANK", "0")), help="local rank for distributed inference")
    parser.add_argument("--pin_memory", type=int, default=0, help="whether to pinned CPU memory for ZeRO offloading")
    parser.add_argument("--cache_dir", type=str, default=".", help="cache dir for model name")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32", "fp64"])
    parser.add_argument("--attn_impl", type=str, default="mem_efficient", choices=["eager", "flash_attention_2", "math", "mem_efficient", "cudnn"], help="torch attention implementation")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--res_folder", type=str, default=".", help="path to store benchmarking results")
    parser.add_argument("--show_chunk_time", type=str, default="y", help="y: measure per chunk time cost; n: no measure")
    
    args = parser.parse_args()

    gc.collect()

    # fix random seed
    fix_randomness_determinism(seed=args.seed, deterministic=False)

    # use monkey patching to overwrite transformers Llama
    assert args.attn_impl != "eager", "no support for eager-based chunked PEFT, " + \
                             "as it's used for correctness check and can be replaced by math..."
    open_performance_mode()

    # timer records
    forward_timings:   List[float] = [ ]
    total_timings:     List[float] = [ ]
    fwd_chunk_timings: List[List[float]] = [ ]
    bwd_chunk_timings: List[List[float]] = [ ]

    # create dataloader (iterate over the same data)
    dataloader_iter = load_data(
        model_name=args.model_name, 
        dataset_name=args.dataset_name, 
        cache_dir=args.cache_dir, 
        seq_len=args.seq_len, 
        batch_size=1, 
        pin_memory=bool(args.pin_memory),
        shuffle=False,
    )

    # create model
    target_modules: List[str] = ["q_proj", "k_proj", "v_proj"]
    model = create_chunk_model(
        model_name=args.model_name,
        pin_memory=bool(args.pin_memory),
        cache_dir=args.cache_dir,
        local_rank=args.local_rank,
        dtype=get_dtype(args.dtype),
        attn_impl=args.attn_impl,
        target_modules=target_modules,
    )
    
    # create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # run chunked peft
    run_chunked_peft_holistic(
        model=model, 
        optimizer=optimizer,
        dataloader_iter=dataloader_iter,
        num_run_steps=args.trials,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        chunk_size=args.chunk_size,
        local_rank=args.local_rank,
        forward_timings=forward_timings,
        total_timings=total_timings,
        show_chunk_time=args.show_chunk_time,
        fwd_chunk_timings=fwd_chunk_timings,
        bwd_chunk_timings=bwd_chunk_timings,
    )
    # get results
    assert args.warmup < args.trials, "warm-up steps should be smaller than total steps..."
    log_str: str = get_stat_str(
        model_name=args.model_name,
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        warmup_steps=args.warmup,
        forward_timings=forward_timings,
        total_timings=total_timings,
        fwd_chunk_timings=fwd_chunk_timings,
        bwd_chunk_timings=bwd_chunk_timings,
    )
    fname: str = f"chunk_{args.dtype}_seqlen{args.seq_len}_chunksize{args.chunk_size}_trials{args.trials}.txt"
    if not os.path.exists(args.res_folder):
        os.makedirs(args.res_folder)
    with open(args.res_folder + "/" + fname, "a+") as fp:
        seed_str: str = f"[seed: {args.seed}]\n"
        fp.write(seed_str + log_str + '\n')