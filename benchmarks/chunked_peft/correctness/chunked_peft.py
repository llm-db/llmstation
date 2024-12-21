import os
import gc
import argparse
import itertools
from collections import OrderedDict
from typing import List, Dict, Tuple, Union, Optional

import torch
from tqdm import tqdm
from transformers import (
    DynamicCache,
)

from benchmarks.chunked_peft.data_utils import (
    load_data,
)

from benchmarks.chunked_peft.chunk_utils import (
    ChunkCache,
    printer,
    INVALID_TARGET,
    fix_determinism,
    fixed_cross_entropy,
    get_learnable_param_grad,
    create_chunk_model,
    get_dtype,
)

from benchmarks.chunked_peft.monkey_patching.monkey_patching import open_correctness_mode


def chunked_peft_forward_naive(model: torch.nn.Module, 
                               inputs: Union[List[int], Dict],
                               chunk_size: int,
                               ) -> Tuple[List[torch.nn.Module], torch.Tensor]:
    assert chunk_size >= 1, "chunk size should be a postive integer"
    num_total_tokens: int = inputs.input_ids.shape[1]
    num_processed_tokens: int = 0
    past_key_values = DynamicCache().to(device=torch.cuda.current_device())
    num_valid_tokens: torch.Tensor = (inputs.labels != INVALID_TARGET).sum(dim=-1, keepdim=True)\
                                                                      .to(device=torch.cuda.current_device()) - 1
    losses: List[torch.nn.Module] = []
    logits: List[torch.Tensor] = []
    while num_processed_tokens < num_total_tokens:
        num_cur_step: int = min(chunk_size, 
                                num_total_tokens-num_processed_tokens)
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
        
        # TODO: check grad ckpt's implications on performance
        # model.gradient_checkpointing_enable(dict(use_reentrant=False))
        outputs = model(**cur_inputs)

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
        
        # NOTE: here we only consider batch_size = 1
        shift_targets = shift_targets.view(-1)
        shift_logits  = shift_logits.view(-1, shift_logits.shape[-1])
        shift_targets = shift_targets.to(shift_logits.device)
        cur_loss = fixed_cross_entropy(shift_logits, shift_targets, 1, INVALID_TARGET)
        
        # cur_loss: torch.Tensor = (loss_fn(shift_logits.permute(0, 2, 1), 
        #                                   shift_targets)).sum(dim=-1, keepdim=True)
        cur_loss /= num_valid_tokens.item() # sum of avg-token-loss in current chunk
        losses.append(cur_loss)
        logits.append(outputs.logits)
        num_processed_tokens += num_cur_step

    whole_logits = torch.cat(logits, dim=1)   
    return losses, whole_logits


def chunked_peft_forward_separate(model: torch.nn.Module, 
                                  inputs: Union[List[int], Dict],
                                  chunk_size: int,
                                  ) -> Tuple[List[torch.nn.Module], torch.Tensor]:
    assert chunk_size >= 1, "chunk size should be a postive integer"
    num_total_tokens: int = inputs.input_ids.shape[1]
    num_processed_tokens: int = 0
    past_key_values = ChunkCache(attn_impl=model.config._attn_implementation).to(device=torch.cuda.current_device())
    num_valid_tokens: torch.Tensor = (inputs.labels != INVALID_TARGET).sum(dim=-1, keepdim=True)\
                                                                      .to(device=torch.cuda.current_device()) - 1
    losses: List[torch.nn.Module] = []
    logits: List[torch.Tensor] = []
    while num_processed_tokens < num_total_tokens:
        num_cur_step: int = min(chunk_size, 
                                num_total_tokens-num_processed_tokens)
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
        
        # TODO: check grad ckpt's implications on performance
        # model.gradient_checkpointing_enable(dict(use_reentrant=False))
        outputs = model(**cur_inputs)


        # print(f"right after a model call, len of key_grads_cache: {len(outputs.past_key_values.key_grads_cache)}")

        # NOTE: detach prev tensor from the computation graph of new chunks
        # past_key_values = ChunkCache(prev_cache=outputs.past_key_values)
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
        
        # NOTE: here we only consider batch_size = 1
        # TODO: extend to batch_size > 1
        shift_targets = shift_targets.view(-1)
        shift_logits  = shift_logits.view(-1, shift_logits.shape[-1])
        shift_targets = shift_targets.to(shift_logits.device)
        cur_loss = fixed_cross_entropy(shift_logits, shift_targets, 1, INVALID_TARGET)
        
        # cur_loss: torch.Tensor = (loss_fn(shift_logits.permute(0, 2, 1), 
        #                                   shift_targets)).sum(dim=-1, keepdim=True)
        cur_loss /= num_valid_tokens.item() # sum of avg-token-loss in current chunk
        losses.append(cur_loss)
        logits.append(outputs.logits)
        num_processed_tokens += num_cur_step

    whole_logits = torch.cat(logits, dim=1)   
    return losses, whole_logits


def chunked_peft_backward(model: torch.nn.Module,
                          losses: List[torch.nn.Module],
                          check_grad: bool = False) -> None:
    printer.print(f"bwd pass in backward chunk order")
    n: int = len(losses)
    for idx, loss in enumerate(losses[::-1]):
        printer.print(f"bwd pass w. chunk: {n-idx-1}")
        loss.backward(retain_graph=(False if idx == n-1 else True))
        printer.print(f"grad of last layer v.proj: {model.base_model.        \
                                                    model.model.layers[-1].  \
                                                    self_attn.v_proj.lora_B. \
                                                    default.weight.grad}")

def chunked_peft_backward_whole(model: torch.nn.Module,
                                losses: List[torch.nn.Module],
                                check_grad: bool = False) -> None:
    printer.print(f"bwd pass w. whole loss")
    loss = sum(losses)
    loss.backward()
    printer.print(f"grad of last layer v.proj: {model.base_model.        \
                                                model.model.layers[-1].  \
                                                self_attn.v_proj.lora_B. \
                                                default.weight.grad}")

def chunked_peft_backward_forward_order(model: torch.nn.Module,
                                        losses: List[torch.nn.Module],
                                        check_grad: bool = False) -> None:
    printer.print(f"bwd pass in forward chunk order")
    n: int = len(losses)
    for idx, loss in enumerate(losses):
        printer.print(f"bwd pass w. chunk: {idx} (forward chunk order)")
        loss.backward(retain_graph=(False if idx == n-1 else True))
        printer.print(f"grad of last layer v.proj: {model.base_model.        \
                                                    model.model.layers[-1].  \
                                                    self_attn.v_proj.lora_B. \
                                                    default.weight.grad}")

def run_chunked_peft_step(model: torch.nn.Module,
                          optimizer: torch.optim.Optimizer,
                          step: int,
                          inputs: Union[List[int], Dict],
                          do_backward: bool,
                          gradient_accumulation_steps: int,
                          chunk_size: int,
                          local_rank: int,
                          check_grad: bool = False,
                          chunk_impl: str = "separate",
                          backward_whole: bool = False,
                          backward_order: bool = True,
                          ) -> Tuple[torch.tensor, torch.tensor, Optional[OrderedDict]]:
    torch.cuda.set_device(local_rank)
    
    if chunk_impl == "naive":
        losses, logits = chunked_peft_forward_naive(model=model, inputs=inputs, chunk_size=chunk_size)
    elif chunk_impl == "separate":
        losses, logits = chunked_peft_forward_separate(model=model, inputs=inputs, chunk_size=chunk_size)
    else:
        raise NotImplementedError()

    torch.cuda.synchronize()
    result = (sum(losses), logits, )
    if do_backward:
        if backward_whole:
            chunked_peft_backward_whole(model=model, losses=losses, check_grad=check_grad)
        else:
            if backward_order:
                chunked_peft_backward(model=model, losses=losses, check_grad=check_grad)
            else:
                chunked_peft_backward_forward_order(model=model, losses=losses, check_grad=check_grad)
        if check_grad:
            result += (get_learnable_param_grad(model), )
        if step % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
    torch.cuda.synchronize()
    return result

def run_chunked_peft_holistic(model: torch.nn.Module, 
                              optimizer: torch.optim.Optimizer,
                              dataloader_iter: itertools.cycle, 
                              num_run_steps: int,
                              do_backward: bool,
                              gradient_accumulation_steps: int,
                              chunk_size: int,
                              local_rank: int,
                              check_grad: bool = False,
                              chunk_impl: str = "separate",
                              backward_whole: bool = False,
                              backward_order: bool = True,
                              ) -> None:
    for step, inputs in tqdm(dataloader_iter):
        run_chunked_peft_step(model, optimizer, step, inputs,
                              do_backward=do_backward,
                              gradient_accumulation_steps=gradient_accumulation_steps,
                              chunk_size=chunk_size,
                              local_rank=local_rank,
                              check_grad=check_grad,
                              chunk_impl=chunk_impl,
                              backward_whole=backward_whole,
                              backward_order=backward_order,)
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
    parser.add_argument("--attn_impl", type=str, default="flash_attention_2", choices=["eager", "flash_attention_2", "math", "mem_efficient"], help="torch attention implementation")
    parser.add_argument("--chunk_impl", type=str, default="separate", choices=["naive", "separate"], help="chunked BWD implementation")
    parser.add_argument("--print_out", type=str, default="y", help="y: print info; n: no show")
    args = parser.parse_args()

    gc.collect()

    # fix random seed
    fix_determinism()

    # open printer
    printer.open() if args.print_out == "y" else printer.close()

    # use monkey patching to overwrite transformers Llama
    assert args.attn_impl != "eager", "no support for eager-based chunked PEFT, " + \
                             "as it's used for correctness check and can be replaced by math..."
    open_correctness_mode()

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
                               local_ranks=args.local_rank,
                               dtype=get_dtype(args.dtype),
                               attn_impl=args.attn_impl,)
    
    # FIXME: add grad print
    # add_print_backward_hooks(model)

    # create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    # run chunked peft
    run_chunked_peft_holistic(model=model, 
                              optimizer=optimizer,
                              dataloader_iter=dataloader_iter,
                              num_run_steps=args.trials,
                              do_backward=True,
                              gradient_accumulation_steps=args.gradient_accumulation_steps,
                              chunk_size=args.chunk_size,
                              local_rank=args.local_rank,
                              check_grad=True,
                              chunk_impl=args.chunk_impl,
                              backward_whole=False,
                              backward_order=True,)