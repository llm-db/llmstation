#!/bin/bash
dtypes=("bf16" "fp16")
# seq_lens=(512 1024)
seq_lens=(1024)
# seeds=(39 40 41 42 43 44 45)
seeds=(42)
for dtype in ${dtypes[@]}
do
       echo current dtype: ${dtype}
       for seq_len in ${seq_lens[@]}
       do
              chunk_size=${seq_len}
              echo current chunk_size: ${chunk_size}
              for seed in ${seeds[@]}
              do
              echo current seed: $seed
              python -m benchmarks.chunked_peft.performance.whole_peft \
                     --model_name /pub/scratch/yanghao/models/huggingface/meta-llama/Llama-3.1-8B \
                     --dataset_name sordonia/flan-10k-flat \
                     --trials 20 \
                     --batch_size 1 \
                     --gradient_accumulation_steps 1 \
                     --seq_len ${seq_len} \
                     --chunk_size ${chunk_size} \
                     --local_rank 2 \
                     --pin_memory 0 \
                     --quant_bits 16 \
                     --quant_group_size 64 \
                     --cache_dir /pub/scratch/yanghao/datasets/ \
                     --attn_impl mem_efficient \
                     --dtype $dtype \
                     --seed $seed \
                     --res_folder results/naive_chunk/performance/ \
                     --print_out n
              done
       done
done