#!/bin/bash

python -m benchmarks.naive_chunk.correctness.chunked_peft \
       --model_name /pub/scratch/yanghao/models/huggingface/Qwen/Qwen2.5-1.5B \
       --dataset_name yahma/alpaca-cleaned \
       --trials 100 \
       --batch_size 1 \
       --gradient_accumulation_steps 1 \
       --seq_len 256 \
       --chunk_size 128 \
       --local_rank 1 \
       --pin_memory 0 \
       --quant_bits 64 \
       --quant_group_size 64 \
       --cache_dir /pub/scratch/yanghao/datasets/ \
       --dtype fp64 \
       --print_out n
