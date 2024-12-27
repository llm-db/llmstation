#!/bin/bash
dtypes=("bf16")
seq_lens=(1024)
seeds=(42)
for dtype in ${dtypes[@]}
do
       echo current dtype: ${dtype}
       for seq_len in ${seq_lens[@]}
       do
              echo current sequence length: ${seq_len}
              for seed in ${seeds[@]}
              do
              echo current seed: $seed
              python -m benchmarks.chunked_peft.whole_peft_performance_measure \
                     --model_name /pub/scratch/yanghao/models/huggingface/meta-llama/Llama-3.1-8B \
                     --dataset_name sordonia/flan-10k-flat \
                     --trials 30 \
                     --warmup 10 \
                     --seq_len ${seq_len} \
                     --local_rank 0 \
                     --cache_dir /pub/scratch/yanghao/datasets/ \
                     --attn_impl mem_efficient \
                     --dtype $dtype \
                     --seed $seed \
                     --res_folder results/mea_performance/
              done
       done
done