import gc
import argparse
import itertools
from collections import OrderedDict
from typing import List, Tuple, Optional, Union, Dict

import torch
from tqdm import tqdm

from benchmarks.chunked_peft.data_utils import (
    load_data,
)

from benchmarks.chunked_peft.chunk_utils import (
    ChunkCache,
    INVALID_TARGET,
    fixed_cross_entropy,
    fix_randomness_determinism,
    copy_input,
    create_chunk_model,
    copy_models,
    check_two_same_models,
    get_learnable_param_grad,
    get_dtype,
)

from benchmarks.chunked_peft.monkey_patching import open_correctness_mode


### whole peft

def whole_peft_forward(
    model: torch.nn.Module, 
    inputs: Union[List[int], Dict],
) -> Tuple[List[torch.nn.Module], torch.Tensor]:
    inputs.to(torch.cuda.current_device())
    outputs = model(**inputs, use_cache=False)
    return outputs.loss, outputs.logits

def whole_peft_backward(loss: torch.nn.Module) -> None:
    loss.backward()

def run_whole_peft_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    inputs: Union[List[int], Dict],
    gradient_accumulation_steps: int,
    local_rank: int,
    check_grad: bool = True,
) -> Tuple[torch.tensor, torch.tensor, Optional[OrderedDict]]:
    torch.cuda.set_device(local_rank)
    loss, logits = whole_peft_forward(model=model, inputs=inputs)
    torch.cuda.synchronize()
    result = (loss, logits, )
    whole_peft_backward(loss=loss)
    if check_grad:
        result += (get_learnable_param_grad(model), )
    if step % gradient_accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
    torch.cuda.synchronize()
    return result

### chunked peft

def chunked_peft_forward(
    model: torch.nn.Module, 
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
        cur_inputs: Dict = {
            "input_ids": cur_input_ids.to(device=torch.cuda.current_device()),
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
        
        # NOTE: here we only consider batch_size = 1
        shift_targets = shift_targets.view(-1)
        shift_logits  = shift_logits.view(-1, shift_logits.shape[-1])
        shift_targets = shift_targets.to(shift_logits.device)
        cur_loss = fixed_cross_entropy(shift_logits, shift_targets, 1, INVALID_TARGET)
        
        cur_loss /= num_valid_tokens.item() # sum of avg-token-loss in current chunk
        losses.append(cur_loss)
        logits.append(outputs.logits)
        num_processed_tokens += num_cur_step

    whole_logits = torch.cat(logits, dim=1)   
    return losses, whole_logits

def chunked_peft_backward(losses: List[torch.nn.Module]) -> None:
    for loss in losses[::-1]:
        loss.backward(retain_graph=False)

def run_chunked_peft_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    inputs: Union[List[int], Dict],
    gradient_accumulation_steps: int,
    chunk_size: int,
    local_rank: int,
    check_grad: bool = True,
) -> Tuple[torch.tensor, torch.tensor, Optional[OrderedDict]]:
    torch.cuda.set_device(local_rank)
    
    losses, logits = chunked_peft_forward(model=model, inputs=inputs, chunk_size=chunk_size)

    result = (sum(losses), logits, )
    
    chunked_peft_backward(losses=losses)
    
    if check_grad:
        result += (get_learnable_param_grad(model), )
    
    if step % gradient_accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
    
    return result


### result check functions

def check_results(
    res_tuple_1, res_tuple_2, 
    tolerance: float = 1e-9, 
    check_grad: bool = True, 
    grad_tolerance: float = 1e-9,
):
    loss_1,   loss_2   = res_tuple_1[0], res_tuple_2[0]
    logits_1, logits_2 = res_tuple_1[1], res_tuple_2[1]
    assert torch.allclose(loss_1.cpu(), loss_2.cpu(), atol=tolerance)
    assert torch.allclose(logits_1.cpu(), logits_2.cpu(), atol=tolerance, rtol=tolerance)
    if check_grad:
        grad_dict_1, grad_dict_2 = res_tuple_1[2], res_tuple_2[2]
        for name, _ in reversed(list(grad_dict_1.items())):
            assert torch.allclose(grad_dict_1[name].cpu(), grad_dict_2[name].cpu(), 
                                  atol=grad_tolerance, rtol=grad_tolerance), \
                    f"[{name}] model 1 grad: {grad_dict_1[name][:-1,:]}; model 2 grad: {grad_dict_2[name][:-1,:]}"

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

def run_and_check_results(
    models: Tuple[torch.nn.Module, torch.nn.Module], 
    optimizers: Tuple[torch.optim.Optimizer,torch.optim.Optimizer],
    dataloader_iter: itertools.cycle, 
    num_run_steps: int,
    gradient_accumulation_steps: int,
    chunk_size: int,
    local_ranks: Tuple[int, int],
    tolerances: Tuple[float, float],
    check_grad: bool = True,
) -> List[Tuple[torch.tensor, torch.tensor, Optional[OrderedDict]]]:
    results: List[Tuple[torch.tensor, torch.tensor, Optional[OrderedDict]]] = []
    valid_step: int = 0
    for step, inputs in tqdm(dataloader_iter):
        # first: whole
        whole_inputs = copy_input(inputs, local_rank=local_ranks[0])
        whole_result = run_whole_peft_step(
            models[0], optimizers[0], valid_step, whole_inputs,
            gradient_accumulation_steps=gradient_accumulation_steps,
            local_rank=local_ranks[0],
            check_grad=check_grad
        )
        # second: chunk
        chunk_inputs = copy_input(inputs, local_rank=local_ranks[1])
        chunk_result = run_chunked_peft_step(
            models[1], optimizers[1], valid_step, chunk_inputs,
            gradient_accumulation_steps=gradient_accumulation_steps,
            chunk_size=chunk_size,
            local_rank=local_ranks[1],
            check_grad=check_grad,
        )
        # check result
        check_results(
            whole_result, chunk_result, 
            tolerance=tolerances[0], 
            check_grad=check_grad, 
            grad_tolerance=tolerances[1]
        )
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
    parser.add_argument("--cache_dir", type=str, default=".", help="cache dir for model name")
    parser.add_argument("--dtype", type=str, default="fp64", choices=["bf16", "fp32", "fp64"])
    parser.add_argument("--attn_impl", type=str, default="mem_efficient", choices=["eager", "flash_attention_2", "math", "mem_efficient", "cudnn"], help="torch attention implementation")
    parser.add_argument("--tolerances", type=float, nargs='+', help="2-element tuple: first is loss/logits tolerance; second is grads tolerance")
    
    args = parser.parse_args()

    gc.collect()

    # fix random seed
    fix_randomness_determinism()

    # use monkey patching to overwrite transformers Llama
    assert args.attn_impl != "eager", "no support for eager-based chunked PEFT, " + \
                             "as it's used for correctness check and can be replaced by math..."
    open_correctness_mode()

    # create dataloader
    dataloader_iter = load_data(
        model_name=args.model_name, 
        dataset_name=args.dataset_name, 
        cache_dir=args.cache_dir, 
        seq_len=args.seq_len, 
        batch_size=1, 
        pin_memory=bool(args.pin_memory)
    )
    # create model: 0 - eager mode [standard attn], 1 - assigned attn impl
    target_modules: List[str] = ["q_proj", "k_proj", "v_proj"]
    models: List[torch.nn.Module] = [ None, None ]
    models[0] = create_chunk_model(
        model_name=args.model_name,
        pin_memory=bool(args.pin_memory),
        cache_dir=args.cache_dir,
        local_rank=args.local_ranks[0],
        dtype=get_dtype(args.dtype),
        attn_impl="eager",
        target_modules=target_modules
    )
    models[1] = create_chunk_model(
        model_name=args.model_name,
        pin_memory=bool(args.pin_memory),
        cache_dir=args.cache_dir,
        local_rank=args.local_ranks[1],
        dtype=get_dtype(args.dtype),
        attn_impl=args.attn_impl,
        target_modules=target_modules
    )
    copy_models(models[0], models[1])
    check_two_same_models(models[0], models[1])

    # create optimizer
    optimizers = (torch.optim.AdamW(models[0].parameters(), lr=3e-4), \
                  torch.optim.AdamW(models[1].parameters(), lr=3e-4),)
    # run chunked peft
    run_and_check_results(
        models=models, 
        optimizers=optimizers,
        dataloader_iter=dataloader_iter,
        num_run_steps=args.trials,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        chunk_size=args.chunk_size,
        tolerances=tuple(args.tolerances),
        local_ranks=args.local_ranks,
        check_grad=True,
    )