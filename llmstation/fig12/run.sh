#!/bin/bash
# $1 - download dir

if [[ $# -lt 1 ]]; then
    echo "Too few arguments. "
    echo "Usage $0 <download dir>"
    exit
fi

download_dir=$1
forward_tasklets=32
forward_wait=0
backward_tasklets=32
backward_wait=0

vllm_server_log=fig12_vllm_server.log
vllm_client_log=fig12_vllm_client.log
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
init_secs=300
echo "Wait $init_secs seconds for vLLM initialization before benchmarking."
echo "You can increase/decrease the time according to your environments"
sleep $init_secs

kill ${vllm_pid}
echo quit | nvidia-cuda-mps-control
exit;
