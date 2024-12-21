#!/bin/bash

python -m benchmarks.chunked_peft.correctness.correctness_check \
       --model_name /pub/scratch/yanghao/models/huggingface/Qwen/Qwen2.5-1.5B \
       --dataset_name yahma/alpaca-cleaned \
       --trials 1 \
       --batch_size 1 \
       --gradient_accumulation_steps 1 \
       --seq_len 256 \
       --chunk_size 128 \
       --tolerances 1e-9 1e-9 \
       --local_ranks 2 3 \
       --pin_memory 0 \
       --quant_bits 64 \
       --quant_group_size 64 \
       --cache_dir /pub/scratch/yanghao/datasets/ \
       --dtype bf16 \
       --attn_impl flash \
       --print_out y
