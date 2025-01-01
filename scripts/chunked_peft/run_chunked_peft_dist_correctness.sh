#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1 \
torchrun --nnode=1 --nproc-per-node=2 \
         -m benchmarks.chunked_peft.lora_ft_dist_chunked \
         --config benchmarks/chunked_peft/configs/llama2_7b_lora_dist_chunked_correctness.yaml \
         max_steps_per_epoch=30 \
         tokenizer.max_seq_len=512 \
         chunk_size=256 \
         attn_impl=math