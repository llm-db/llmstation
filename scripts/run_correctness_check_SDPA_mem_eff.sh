#!/bin/bash

python -m benchmarks.chunked_peft.chunked_peft_correctness_check  \
       --model_name /pub/scratch/yanghao/models/huggingface/Qwen/Qwen2.5-1.5B \
       --dataset_name yahma/alpaca-cleaned \
       --trials 15 \
       --seq_len 256 \
       --chunk_size 128 \
       --tolerances 5e-4 5e-4 \
       --local_ranks 0 1 \
       --cache_dir /pub/scratch/yanghao/datasets/ \
       --dtype fp32 \
       --attn_impl mem_efficient
