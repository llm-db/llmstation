#!/bin/bash
dtypes=("fp16")
seq_lens=(1024)
seeds=(42)
for dtype in ${dtypes[@]}
do
       echo current dtype: ${dtype}
       for seq_len in ${seq_lens[@]}
       do
              echo current seq_len: ${seq_len}
              if [ ${seq_len} -eq 1024 ]; then
                     # chunk_sizes=(1 2 4 8 16 32 64 128 256 512 1024)
                     chunk_sizes=(128)
              else
                     # chunk_sizes=(1 2 4 8 16 32 64 128 256 512)
                     chunk_sizes=(128)
              fi

              echo chunk_sizes: ${chunk_sizes[@]}
              for chunk_size in ${chunk_sizes[@]}
              do
                     echo current chunk_size: ${chunk_size}
                     for seed in ${seeds[@]}
                     do
                     echo current seed: $seed
                     python -m benchmarks.chunked_peft.chunked_peft_performance_measure \
                            --model_name /pub/scratch/yanghao/models/huggingface/meta-llama/Llama-3.1-8B \
                            --dataset_name sordonia/flan-10k-flat \
                            --trials 30 \
                            --warmup 10 \
                            --seq_len ${seq_len} \
                            --chunk_size ${chunk_size} \
                            --local_rank 0 \
                            --cache_dir /pub/scratch/yanghao/datasets/ \
                            --attn_impl flash_attention_2 \
                            --dtype $dtype \
                            --seed $seed \
                            --res_folder results/fa_performance/ \
                            --show_chunk_time y
                     done
              done
       done
done