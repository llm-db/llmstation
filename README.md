# Chunked Fine-Tuning / PEFT

One thing to note is that even in the latest stable version, `torch 2.5.1`, SDPA flash attn cannot be used since in chunked fine-tuning, (1) the causal mask fed to `sdpa` kernel is not none, (2) and for the non mask case, `q_len = k_len` is required, where `q_len` $\neq$ `k_len` starting from the 2nd chunk. Some discussions can be seen in https://github.com/pytorch/torchtune/issues/1380, and https://huggingface.co/docs/optimum/bettertransformer/overview. Therefore, to use `FA2`, please install `flash_attn` and set `attn_impl` as `flash_attention_2`.

## Environment Setup

```[shell]
conda create -n FineInferChunked python=3.12
conda activate FineInferChunked
pip install -r requirements.txt
```

## Correctness Verification

NOTE: Considering numerical stability issues, SDPA math which supports fp64 is recommended for correctness verification, as SDPA attn mode can be seamlessly switched by Pytorch level `enable/disable`. 

- Eager [standard attn, fp64] v.s. Torch SDPA [math, fp64]

    ```[shell]
    python -m benchmarks.chunked_peft.chunked_peft_correctness_check \
       --model_name <YOUR QWEN2.5-1.5B PATH> \
       --dataset_name yahma/alpaca-cleaned \
       --trials 100 \
       --seq_len 256 \
       --chunk_size 16 \
       --tolerances 1e-9 1e-9 \
       --local_ranks 0 1 \
       --cache_dir <YOUR DATASET CACHE DIR> \
       --dtype fp64 \
       --attn_impl math
    ```

- Eager [standard attn, fp32] v.s. Torch SDPA [memory efficient attn, fp32]

    This can be done, but only with small `trials` and high `tolerances`, as the accumulated numerical errors w.r.t. `fp32` go larger through iterations.

    ```
    python -m benchmarks.chunked_peft.chunked_peft_correctness_check \
       --model_name <YOUR QWEN2.5-1.5B PATH> \
       --dataset_name yahma/alpaca-cleaned \
       --trials 15 \
       --seq_len 256 \
       --chunk_size 128 \
       --tolerances 5e-4 5e-4 \
       --local_ranks 0 1 \
       --cache_dir <YOUR DATASET CACHE DIR> \
       --dtype fp32 \
       --attn_impl mem_efficient
    ```

## Performance Measurement

- Torch SDPA [no-chunk, memory efficient attn, bf16]

    ```
    python -m benchmarks.chunked_peft.whole_peft_performance_measure \
        --model_name <YOUR LLAMA-3.1-8B PATH> \
        --dataset_name sordonia/flan-10k-flat \
        --trials 30 \
        --warmup 10 \
        --seq_len 1024 \
        --local_rank 0 \
        --cache_dir <YOUR DATASET CACHE DIR> \
        --attn_impl mem_efficient \
        --dtype bf16 \
        --seed 42 \
        --res_folder results/mea_performance/
    ```

- Torch SDPA [chunked, memory efficient attn, bf16]

    ```
    python -m benchmarks.chunked_peft.chunked_peft_performance_measure \
        --model_name <YOUR LLAMA-3.1-8B PATH> \
        --dataset_name sordonia/flan-10k-flat \
        --trials 30 \
        --warmup 10 \
        --seq_len 1024 \
        --chunk_size 128 \
        --local_rank 0 \
        --cache_dir <YOUR DATASET CACHE DIR> \
        --attn_impl mem_efficient \
        --dtype bf16 \
        --seed 42 \
        --res_folder results/mea_performance/ \
        --show_chunk_time y
    ```

- Flash Attn [chunked, fp16] (requires `flash_attn` installed)

    ```
    python -m benchmarks.chunked_peft.chunked_peft_performance_measure \
        --model_name <YOUR LLAMA-3.1-8B PATH> \
        --dataset_name sordonia/flan-10k-flat \
        --trials 30 \
        --warmup 10 \
        --seq_len 1024 \
        --chunk_size 128 \
        --local_rank 0 \
        --cache_dir <YOUR DATASET CACHE DIR> \
        --attn_impl flash_attention_2 \
        --dtype fp16 \
        --seed 42 \
        --res_folder results/fa_performance/ \
        --show_chunk_time y
    ```