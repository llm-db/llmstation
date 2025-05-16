#!/bin/bash
# $1 - download dir
# $2 - request rate
# $3 - CUDA_MPS_ACTIVE_THREAD_PERCENTAGE

if [[ $# -lt 3 ]]; then
    echo "Too few arguments. "
    echo "Usage $0 <download dir> <request rate> <CUDA_MPS_ACTIVE_THREAD_PERCENTAGE>"
    exit
fi

download_dir=$1
request_rate=$2
mps_percent=$3
num_prompts=$((request_rate*200))

vllm_server_log=fig8a_vllm_server.log
vllm_client_log=fig8a_vllm_client.log
torchtune_log=fig8a_torchtune.log
torchtune_output_dir=torchtune_output

get_val() {
  grep "^$1:" ../../python/lora_modules.txt | cut -d: -f2-
}
lora_adapter="ETH-LLMSys/Meta-Llama-3.1-8B-LoRA"
lora_path=$(get_val $lora_adapter)
rm -rf *.log
rm -rf "./${torchtune_output_dir}"

export CUDA_VISIBLE_DEVICES=0,1
export CUDA_MPS_PIPE_DIRECTORY=./nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=./nvidia-log
nvidia-cuda-mps-control -d

nohup vllm serve meta-llama/Meta-Llama-3.1-8B \
	-tp=2 --download_dir $download_dir \
	--disable-log-requests --max-model-len=2048 --gpu-memory-utilization=0.45 \
	--enable-lora --max-loras 4 --max-lora-rank=8 \
	--lora-modules chat-lora-0=$lora_path \
			chat-lora-1=$lora_path \
			chat-lora-2=$lora_path \
			chat-lora-3=$lora_path \
	> $vllm_server_log 2>&1 &
vllm_pid=$!
echo "vLLM runs in process ${vllm_pid}"
init_secs=120
echo "Wait $init_secs seconds for vLLM initialization before benchmarking."
echo "You can increase/decrease the time according to your environments"
sleep $init_secs


export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$mps_percent
nohup tune run --nproc_per_node 2 lora_finetune_distributed --config llama3_1/8B_lora \
	model.lora_attn_modules=“[q_proj,k_proj,v_proj]” model.lora_rank=8 \
	model.apply_lora_to_mlp=False model.apply_lora_to_output=False \
	dataset._component_=torchtune.datasets.alpaca_cleaned_dataset dataset.packed=True \
	output_dir="./${torchtune_output_dir}" log_every_n_steps=16 \
	checkpointer.checkpoint_dir="${download_dir}/Meta-Llama-3.1-8B" \
	tokenizer.path="${download_dir}/Meta-Llama-3.1-8B/original/tokenizer.model" tokenizer.max_seq_len=512 \
	enable_activation_checkpointing=True \
	batch_size=1 gradient_accumulation_steps=1 max_steps_per_epoch=1024 \
	> $torchtune_log 2>&1 &
torchtune_pid=$!
echo "torchtune runs in process ${torchtune_pid}"
echo "Wait $init_secs seconds for torchtune initialization before benchmarking."
echo "You can increase/decrease the time according to your environments"
sleep $init_secs


latest_torchtune_log=$(ls -t "./${torchtune_output_dir}"/* | head -n1)
start_line=$(wc -l < $latest_torchtune_log)

python ../../python/benchmark_serving.py \
        --backend vllm --model meta-llama/Meta-Llama-3.1-8B \
        --dataset-name sharegpt --dataset-path "$download_dir/ShareGPT_V3_unfiltered_cleaned_split.json" \
	--request-rate $request_rate --num-prompts $num_prompts \
	> $vllm_client_log

end_line=$(wc -l < $latest_torchtune_log)

python ../parse_log.py --vllm-log $vllm_client_log --torchtune-log $latest_torchtune_log \
	--mps-percent $mps_percent --start-line $start_line --end-line $end_line

kill ${vllm_pid}
kill ${torchtune_pid}
echo quit | nvidia-cuda-mps-control
exit;
