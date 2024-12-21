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
                 forward_timings: List[List[float]],
                 total_timings: List[List[float]],
                 ) -> str:
    print(f"Summary:")
    print(f"total_timings = {total_timings}")
    print(f"forward_timings = {forward_timings}")
    
    total_latency: float    = sum(total_timings) / len(total_timings)
    forward_latency: float  = sum(forward_timings) / len(forward_timings)
    backward_latency: float = total_latency - forward_latency

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
    )
    print(log_str)
    return log_str
