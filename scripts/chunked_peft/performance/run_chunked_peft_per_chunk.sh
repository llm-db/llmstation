#!/bin/bash
dtypes=("bf16" "fp16")
# dtypes=("bf16")
# dtypes=("fp16")
# seq_lens=(512 1024)
# seq_lens=(512)
seq_lens=(1024)
# seq_lens=(768)
# seq_lens=(128)
# seeds=(39 40 41 42 43 44 45)
seeds=(42)
for dtype in ${dtypes[@]}
do
       echo current dtype: ${dtype}
       for seq_len in ${seq_lens[@]}
       do
              echo current seq_len: ${seq_len}
              if [ ${seq_len} -eq 1024 ]; then
                     # chunk_sizes=(32 64 128 256 512 1024)
                     chunk_sizes=(512)
              else
                     # chunk_sizes=(32 64 128 256 512)
                     # chunk_sizes=(16)
                     # chunk_sizes=(64)
                     chunk_sizes=(128)
              fi

              echo chunk_sizes: ${chunk_sizes[@]}
              for chunk_size in ${chunk_sizes[@]}
              do
                     echo current chunk_size: ${chunk_size}
                     for seed in ${seeds[@]}
                     do
                     echo current seed: $seed
                     python -m benchmarks.chunked_peft.performance.chunked_peft \
                            --model_name /pub/scratch/yanghao/models/huggingface/meta-llama/Llama-3.1-8B \
                            --dataset_name sordonia/flan-10k-flat \
                            --trials 6 \
                            --batch_size 1 \
                            --gradient_accumulation_steps 1 \
                            --seq_len ${seq_len} \
                            --chunk_size ${chunk_size} \
                            --local_rank 2 \
                            --pin_memory 0 \
                            --quant_bits 16 \
                            --quant_group_size 64 \
                            --cache_dir /pub/scratch/yanghao/datasets/ \
                            --attn_impl flash_attention_2 \
                            --chunk_impl separate \
                            --dtype $dtype \
                            --seed $seed \
                            --res_folder results/naive_chunk/performance/ \
                            --show_chunk_time y \
                            --print_out n
                     done
              done
       done
done