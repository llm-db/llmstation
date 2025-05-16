import argparse
import os
from pathlib import Path
from huggingface_hub import snapshot_download


parser = argparse.ArgumentParser(description="Path to cache huggingface adapters/models")
parser.add_argument(
    "--download-dir",
    type=Path,
    help="Path to cache huggingface datasets/models/adapters")
args = parser.parse_args()

cache_dir = args.download_dir
file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lora_modules.txt")


with open(file_path, 'w', encoding='utf-8') as f:
    lora_path = snapshot_download(repo_id="ETH-LLMSys/Meta-Llama-3.1-8B-LoRA", cache_dir=cache_dir)
    print(f'ETH-LLMSys/Meta-Llama-3.1-8B-LoRA:{lora_path}', file=f)
    lora_path = snapshot_download(repo_id="ETH-LLMSys/Meta-Llama-3.1-8B-LoRA-rank-16", cache_dir=cache_dir)
    print(f'ETH-LLMSys/Meta-Llama-3.1-8B-LoRA-rank-16:{lora_path}', file=f)
    lora_path = snapshot_download(repo_id="ETH-LLMSys/Meta-Llama-3.1-8B-LoRA-rank-32", cache_dir=cache_dir)
    print(f'ETH-LLMSys/Meta-Llama-3.1-8B-LoRA-rank-32:{lora_path}', file=f)
    # lora_path = snapshot_download(repo_id="ETH-LLMSys/Llama-2-13b-hf-lora", cache_dir=cache_dir)
    # print(f'ETH-LLMSys/Llama-2-13b-hf-lora:{lora_path}', file=f)
    # lora_path = snapshot_download(repo_id="ETH-LLMSys/Meta-Llama-3.1-70B-LoRA", cache_dir=cache_dir)
    # print(f'ETH-LLMSys/Meta-Llama-3.1-70B-LoRA:{lora_path}', file=f)
