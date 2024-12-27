#!/bin/bash

python -m benchmarks.chunked_peft.chunked_peft_correctness_check \
       --model_name /pub/scratch/yanghao/models/huggingface/Qwen/Qwen2.5-1.5B \
       --dataset_name yahma/alpaca-cleaned \
       --trials 100 \
       --seq_len 256 \
       --chunk_size 16 \
       --tolerances 1e-9 1e-9 \
       --local_ranks 0 1 \
       --cache_dir /pub/scratch/yanghao/datasets/ \
       --dtype fp64 \
       --attn_impl math
