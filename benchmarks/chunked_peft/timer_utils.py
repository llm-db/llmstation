from typing import List

import torch
from transformers import (
    AutoConfig,
)

from benchmarks.utils import (
    model_bytes,
    write_peft_benchmark_log
)

def get_stat_str(model_name: str,
                 cache_dir: str,
                 batch_size: int,
                 warmup_steps: int,
                 forward_timings: List[float],
                 total_timings: List[float],
                 fwd_chunk_timings: List[List[float]] = [],
                 bwd_chunk_timings: List[List[float]] = [],
                 ) -> str:
    print(f"Summary:")
    print(f"total_timings = {total_timings}")
    print(f"forward_timings = {forward_timings}")
    
    total_latency: float    = sum(total_timings[warmup_steps:]) / len(total_timings[warmup_steps:])
    forward_latency: float  = sum(forward_timings[warmup_steps:]) / len(forward_timings[warmup_steps:])
    backward_latency: float = total_latency - forward_latency

    assert batch_size == 1, "batch size is set as 1 for chunked fine-tuning..."

    fwd_chunk_avg_timings: List[float] = [ ]
    bwd_chunk_avg_timings: List[float] = [ ]
    if len(fwd_chunk_timings) > warmup_steps and len(fwd_chunk_timings[warmup_steps]) > 0:
        num_chunk: int = len(fwd_chunk_timings[warmup_steps])
        num_valid_steps: int = len(fwd_chunk_timings[warmup_steps:])
        fwd_chunk_avg_timings: List[float] = [ 0.0 ] * num_chunk
        bwd_chunk_avg_timings: List[float] = [ 0.0 ] * num_chunk
        for fwd_list, bwd_list in zip(fwd_chunk_timings[warmup_steps:], 
                                      bwd_chunk_timings[warmup_steps:]):
            for idx, (fwd_chunk_time, bwd_chunk_time) in enumerate(zip(fwd_list, bwd_list)):
                fwd_chunk_avg_timings[idx] += fwd_chunk_time / num_valid_steps / batch_size
                bwd_chunk_avg_timings[idx] += bwd_chunk_time / num_valid_steps / batch_size

    total_throughput: float    = batch_size / total_latency
    forward_throughput: float  = batch_size / forward_latency
    backward_throughput: float = batch_size / backward_latency
    gpu_peak_mem = torch.cuda.max_memory_allocated(torch.device("cuda"))

    config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
    model_size: int = model_bytes(config)
    log_str: str = write_peft_benchmark_log(
        model_size,
        0,
        gpu_peak_mem,
        forward_latency,
        forward_throughput,
        backward_latency,
        backward_throughput,
        total_latency,
        total_throughput,
        warmup_steps=warmup_steps,
        fwd_chunk_avg_timings=fwd_chunk_avg_timings,
        bwd_chunk_avg_timings=bwd_chunk_avg_timings,
    )
    print(log_str)
    return log_str
