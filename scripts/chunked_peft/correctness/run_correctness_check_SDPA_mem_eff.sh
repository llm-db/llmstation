#!/bin/bash

python -m benchmarks.chunked_peft.correctness.correctness_check \
       --model_name /pub/scratch/yanghao/models/huggingface/Qwen/Qwen2.5-1.5B \
       --dataset_name yahma/alpaca-cleaned \
       --trials 20 \
       --batch_size 1 \
       --gradient_accumulation_steps 1 \
       --seq_len 256 \
       --chunk_size 32 \
       --tolerances 5e-4 5e-4 \
       --local_ranks 0 1 \
       --pin_memory 0 \
       --quant_bits 64 \
       --quant_group_size 64 \
       --cache_dir /pub/scratch/yanghao/datasets/ \
       --dtype fp32 \
       --attn_impl mem_efficient \
       --print_out n
