#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1 \
torchrun --nnode=1 --nproc-per-node=2 \
         -m benchmarks.chunked_peft.lora_ft_dist_whole \
         --config benchmarks/chunked_peft/configs/llama2_7b_lora_dist_whole_correctness.yaml \
         max_steps_per_epoch=30 \
         tokenizer.max_seq_len=512 \
         num_output_chunks=2 \
         attn_impl=math