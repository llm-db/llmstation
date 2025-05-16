#!/bin/bash
# $1 - download dir
# $2 - defer in seconds before new prefill

if [[ $# -lt 2 ]]; then
    echo "Too few arguments. "
    echo "Usage $0 <download dir> <fineinfer defer>"
    exit
fi

download_dir=$1
request_rate=0.88
fineinfer_defer=$2
num_prompts=400

vllm_server_log=fig10_vllm_server.log
vllm_client_log=fig10_vllm_client.log
fineinfer_output_dir=$PWD

get_val() {
  grep "^$1:" ../../python/lora_modules.txt | cut -d: -f2-
}
lora_adapter="ETH-LLMSys/Meta-Llama-3.1-8B-LoRA"
lora_path=$(get_val $lora_adapter)
rm -rf *.log

export CUDA_VISIBLE_DEVICES=0,1

nohup vllm serve meta-llama/Meta-Llama-3.1-8B \
	-tp=2 --download_dir $download_dir \
	--disable-async-output-proc --disable-log-requests --max-model-len=2048 --gpu-memory-utilization=0.75 \
	--enable-lora --max-loras 4 --max-lora-rank=8 \
	--enable-fineinfer --fineinfer-defer=$fineinfer_defer --fineinfer-output=$fineinfer_output_dir \
	--lora-modules chat-lora-0=$lora_path \
			chat-lora-1=$lora_path \
			chat-lora-2=$lora_path \
			chat-lora-3=$lora_path \
	> $vllm_server_log 2>&1 &
vllm_pid=$!
echo "vLLM runs in process ${vllm_pid}"
init_secs=240
echo "Wait $init_secs seconds for vLLM initialization before benchmarking."
echo "You can increase/decrease the time according to your environments"
sleep $init_secs


start_line=$(wc -l < "${fineinfer_output_dir}/fineinfer.log")

python ../../python/benchmark_serving_burstgpt.py \
        --backend vllm --model meta-llama/Meta-Llama-3.1-8B \
        --dataset-name sharegpt --dataset-path "$download_dir/ShareGPT_V3_unfiltered_cleaned_split.json" \
	--request-rate $request_rate --num-prompts $num_prompts --download-dir $download_dir \
	> $vllm_client_log

end_line=$(wc -l < "${fineinfer_output_dir}/fineinfer.log")

python ../parse_log.py --vllm-log $vllm_client_log --fineinfer-log "${fineinfer_output_dir}/fineinfer.log" \
	--fineinfer-defer $fineinfer_defer --start-line $start_line --end-line $end_line

kill ${vllm_pid}
exit;
