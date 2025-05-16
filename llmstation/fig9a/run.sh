#!/bin/bash
# $1 - download dir
# $2 - request rate
# $3 - Number of tasklets performed in forward pass before preemption
# $4 - Wait in seconds after preemption in forward pass
# $5 - Number of tasklets performed in backward pass before preemption
# $6 - Wait in seconds after preemption in backward pass

if [[ $# -lt 6 ]]; then
    echo "Too few arguments. "
    echo "Usage $0 <download dir> <request rate> <forward tasklets> <forward wait> <backward tasklets> <backward wait>"
    exit
fi

download_dir=$1
request_rate=$2
forward_tasklets=$3
forward_wait=$4
backward_tasklets=$5
backward_wait=$6
num_prompts=$((request_rate*200))

vllm_server_log=fig9a_vllm_server.log
vllm_client_log=fig9a_vllm_client.log
lms_output_dir=$PWD

get_val() {
  grep "^$1:" ../../python/lora_modules.txt | cut -d: -f2-
}
lora_adapter="ETH-LLMSys/Meta-Llama-3.1-8B-LoRA"
lora_path=$(get_val $lora_adapter)
rm -rf *.log

export CUDA_VISIBLE_DEVICES=0,1
export CUDA_MPS_PIPE_DIRECTORY=./nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=./nvidia-log
nvidia-cuda-mps-control -d

nohup vllm serve meta-llama/Meta-Llama-3.1-8B \
	-tp=2 --download_dir $download_dir \
	--disable-async-output-proc --disable-log-requests --max-model-len=2048 --gpu-memory-utilization=0.75 \
	--enable-lora --max-loras 4 --max-lora-rank=8 \
	--enable-lms --lms-output=$lms_output_dir \
	--lms-forward-tasklets $forward_tasklets --lms-forward-wait $forward_wait \
	--lms-backward-tasklets $backward_tasklets --lms-backward-wait $backward_wait \
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


start_line=$(wc -l < "${lms_output_dir}/lms.log")

python ../../python/benchmark_serving_burstgpt.py \
        --backend vllm --model meta-llama/Meta-Llama-3.1-8B \
        --dataset-name sharegpt --dataset-path "$download_dir/ShareGPT_V3_unfiltered_cleaned_split.json" \
		--request-rate $request_rate --num-prompts $num_prompts --download-dir $download_dir \
	> $vllm_client_log

end_line=$(wc -l < "${lms_output_dir}/lms.log")

python ../parse_log.py --vllm-log $vllm_client_log --lms-log "${lms_output_dir}/lms.log" \
        --lms-forward-tasklets $forward_tasklets --lms-forward-wait $forward_wait \
        --lms-backward-tasklets $backward_tasklets --lms-backward-wait $backward_wait \
		--start-line $start_line --end-line $end_line

kill ${vllm_pid}
echo quit | nvidia-cuda-mps-control
exit;
