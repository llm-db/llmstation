# Distributed Chunked PEFT w/ Torchtune

One thing to note is that [torchtune](https://pytorch.org/torchtune/stable/index.html) supports `sdpa` and `flex attention`, while [Flex Attention](https://arxiv.org/pdf/2412.05496) is a new feature appearing in `torch 2.5.0` which works well in `torch.compile` and still in development. Therefore, here we only consider the `sdpa` kernel. 

Similar to the single-node case, even in the latest stable version, `torch 2.5.1`, SDPA flash attn cannot be used since in chunked fine-tuning, (1) the causal mask fed to sdpa kernel is not none, (2) and for the non mask case, `q_len = k_len` is required, where `q_len` $\neq$ `k_len` starting from the 2nd chunk. Some discussions can be seen in https://github.com/pytorch/torchtune/issues/1380, and https://github.com/pytorch/pytorch/issues/108108.

Hence, we mainly consider the `SDPA memory efficient attention`. For `torch 2.5.0+`, `SDPA cuDNN attention` is also supported.


## Environment Setups

```[shell]
conda create -n ChunkedTrainingDist python=3.12
conda activate ChunkedrainingDist
pip install -r requirements.txt
```

## Download Torchtune Models

```
tune download meta-llama/Llama-2-7b-hf  --output-dir <YOUR LLAMA-2-7B-HF PATH>  --ignore-patterns "*.safetensors"
tune download meta-llama/Llama-2-13b-hf --output-dir <YOUR LLAMA-2-13B-HF PATH> --ignore-patterns "*.safetensors"
```

## Correctness Verification

NOTE: As `torchtune` currently only support `bf16` and `fp32`,  considering numerical stability issues, which is further enlarged by [Fully Sharded Data Parallel (FSDP)](https://www.vldb.org/pvldb/vol16/p3848-huang.pdf), here we use the global loss in consecutive training steps (recorded in the `log`) as the metric to verify the correctness.

- Whole LORA fine-tuning, SDPA math attention, fp32

    ```[shell]
    torchrun --nnode=1 --nproc-per-node=2 \
        -m benchmarks.chunked_peft.lora_ft_dist_whole \
        --config benchmarks/chunked_peft/configs/llama2_7b_lora_dist_whole_correctness.yaml \
        output_dir=<YOUR OUTPUT FOLDER PATH> \
        checkpoint_dir=<YOUR MODEL CKPT PATH> \
        tokenizer.path=<YOUR TOKENIZER PATH> \
        cache_dir=<YOUR DATASET CACHE PATH> \
        max_steps_per_epoch=30 \
        tokenizer.max_seq_len=512 \
        num_output_chunks=2 \
        attn_impl=math
    ```

- Chunked LORA fine-tuning, SDPA math attention, fp32

    ```[shell]
    torchrun --nnode=1 --nproc-per-node=2 \
        -m benchmarks.chunked_peft.lora_ft_dist_chunked \
        --config benchmarks/chunked_peft/configs/llama2_7b_lora_dist_chunked_correctness.yaml \
        output_dir=<YOUR OUTPUT FOLDER PATH> \
        checkpoint_dir=<YOUR MODEL CKPT PATH> \
        tokenizer.path=<YOUR TOKENIZER PATH> \
        cache_dir=<YOUR DATASET CACHE PATH> \
        max_steps_per_epoch=30 \
        tokenizer.max_seq_len=512 \
        chunk_size=256 \
        attn_impl=math
    ```


## Performance Measurement

- Whole LORA fine-tuning, SDPA MEA, bf16

    ```[shell]
    torchrun --nnode=1 --nproc-per-node=2 \
        -m benchmarks.chunked_peft.lora_ft_dist_whole \
        --config benchmarks/chunked_peft/configs/llama2_13b_lora_dist_whole_performance.yaml \
        output_dir=<YOUR OUTPUT FOLDER PATH> \
        checkpoint_dir=<YOUR MODEL CKPT PATH> \
        tokenizer.path=<YOUR TOKENIZER PATH> \
        cache_dir=<YOUR DATASET CACHE PATH> \
        max_steps_per_epoch=30 \
        warmup_steps=10 \
        tokenizer.max_seq_len=1024 \
        num_output_chunks=8 \
        attn_impl=mem_efficient
    ```

- Chunked LORA fine-tuning, SDPA MEA, bf16

    ```[shell]
    torchrun --nnode=1 --nproc-per-node=2 \
        -m benchmarks.chunked_peft.lora_ft_dist_chunked \
        --config benchmarks/chunked_peft/configs/llama2_13b_lora_dist_chunked_performance.yaml \
        output_dir=<YOUR OUTPUT FOLDER PATH> \
        checkpointer.checkpoint_dir=<YOUR MODEL CKPT PATH> \
        tokenizer.path=<YOUR TOKENIZER PATH> \
        dataset.cache_dir=<YOUR DATASET CACHE PATH> \
        max_steps_per_epoch=30 \
        warmup_steps=10 \
        tokenizer.max_seq_len=1024 \
        chunk_size=128 \
        attn_impl=mem_efficient
    ```
