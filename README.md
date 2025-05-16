# ATC '25 Artifact
## Setup
### Software requirements
* Ubuntu 22.04
* Conda
* Python 3.12
* CUDA 12.6
* cmake 3.26
* gcc-13.3.0
* g++-13.3.0

### Hardware requirements
Running these scripts requires 2 GPUs, each with 24 GB HBM.

### Installation
We recommend users to use two conda environments, one for LLMStation and the other for baselines.
Since LLMStation runs on customized PyTorch 2.4.0 and vLLM 0.6.3, we also recommend that users build vanilla PyTorch 2.4.0 and vLLM 0.6.3 from source for baselines,
rather than using pre-built binaries, to avoid performance differences introduced by compilation configurations.
* __Baselines__

Conda environment
```
conda create -n baseline python=3.12.4
conda activate baseline
# conda deactivate
# conda remove --name baseline --all
```
[Install PyTorch 2.4.0 from scratch](https://github.com/pytorch/pytorch/tree/v2.4.0?tab=readme-ov-file#from-source)
```
# conda install -c conda-forge libstdcxx-ng=13
# export CUDA_HOME=/usr/local/cuda-12.6
# export PATH=$CUDA_HOME/bin:$PATH
# export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
git clone -b pytorch-v2.4.0 https://github.com/llm-db/llmstation.git pytorch-v2.4.0
cd pytorch-v2.4.0
git submodule sync
git submodule update --init --recursive
conda install cmake ninja
pip install -r requirements.txt
python -m pip install mkl-include mkl-static
conda install -c pytorch magma-cuda126
make triton
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
MAX_JOBS=32 CMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc CMAKE_CUDA_ARCHITECTURES=native python setup.py develop
```
[Install vLLM 0.6.3 from scratch](https://docs.vllm.ai/en/v0.6.3/getting_started/installation.html#use-an-existing-pytorch-installation)
```
git clone -b vllm-v0.6.3 https://github.com/llm-db/llmstation.git vllm-v0.6.3
cd vllm-v0.6.3
python use_existing_torch.py
pip install -r requirements-build.txt
MAX_JOBS=32 pip install . --no-build-isolation
# git clean -xfd
# MAX_JOBS=32 pip install . --no-build-isolation --no-cache-dir --force-reinstall
```
[Install torchtune 0.4.0](https://github.com/pytorch/torchtune/tree/v0.4.0?tab=readme-ov-file#installation)
```
pip install torchao==0.6.1
pip install torchtune==0.4.0
```
Switch back to transformers 4.45.0
```
# vLLM 0.6.3 may not be compatible with newer versions of transformers
pip install transformers==4.45.0
```

* __LLMStation__

Conda environment
```
conda create -n lms python=3.12.4
conda activate lms
# conda deactivate
# conda remove --name lms --all
```
Install customized PyTorch 2.4.0
```
# conda install -c conda-forge libstdcxx-ng=13
# export CUDA_HOME=/usr/local/cuda-12.6
# export PATH=$CUDA_HOME/bin:$PATH
# export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
git clone -b pytorch-v2.4.0-lms https://github.com/llm-db/llmstation.git pytorch-v2.4.0-lms
cd pytorch-v2.4.0-lms
git submodule sync
git submodule update --init --recursive
conda install cmake ninja
pip install -r requirements.txt
python -m pip install mkl-include mkl-static
conda install -c pytorch magma-cuda126
make triton
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
MAX_JOBS=32 CMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc CMAKE_CUDA_ARCHITECTURES=native python setup.py develop
```
Install customized vLLM 0.6.3
```
git clone -b vllm-v0.6.3-lms https://github.com/llm-db/llmstation.git vllm-v0.6.3-lms
cd vllm-v0.6.3-lms
python use_existing_torch.py
pip install -r requirements-build.txt
MAX_JOBS=32 pip install . --no-build-isolation
# git clean -xfd
# MAX_JOBS=32 pip install . --no-build-isolation --no-cache-dir --force-reinstall
```
Switch back to transformers 4.45.0
```
# vLLM 0.6.3 may not be compatible with newer versions of transformers
pip install transformers==4.45.0
```


### Download datasets, base models and adapters
Many scripts in the benchmark require --download-dir as an input parameter to specific paths for data and models.
We recommend users to store them in the same folder mounted on a fast SSD.
```
cd $HOME/scratch
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
wget https://github.com/HPMLL/BurstGPT/releases/download/v1.1/BurstGPT_without_fails_2.csv
cd ~/llmstation
python python/llama_lora_download.py --download-dir=$HOME/scratch
./vLLM-torchtune/llama_torchtune_download.sh $HOME/scratch
```

### Benchmark
After each run, please check the logs for vLLM (server and client), torch tune, and [NVIDIA MPS](https://docs.nvidia.com/deploy/mps/index.html)
within the same folder to make sure they are all working properly.
In cases where the system cannot meet the SLO no matter how the user tunes the input parameters, the throughput will be reported as zero.
* __vLLM + torchtune__
  - For Figures 8 and 9, by varying the request rate, you can get the results in result.txt within the same folder.
    However, you may find that some configurations violate the SLOs.
    In these cases, you will also need to adjust the CUDA_MPS_ACTIVE_THREAD_PERCENTAGE until the results meet the SLOs.
  - For Figure 10, you only need to adjust CUDA_MPS_ACTIVE_THREAD_PERCENTAGE until the PEFT throughput is the same as points on the X-axis.
  - For Figure 13, you need to adjust LoRA distribution, LoRA rank, and CUDA_MPS_ACTIVE_THREAD_PERCENTAGE.
```
cd vLLM-torchtune
conda activate baseline
cd fig8a
# ./run.sh <download dir> <request rate> <CUDA_MPS_ACTIVE_THREAD_PERCENTAGE>
./run.sh $HOME/scratch 1 80
cd fig9a
# ./run.sh <download dir> <request rate> <CUDA_MPS_ACTIVE_THREAD_PERCENTAGE>
./run.sh $HOME/scratch 1 80
cd fig10
# ./run.sh <download dir> <CUDA_MPS_ACTIVE_THREAD_PERCENTAGE>
./run.sh $HOME/scratch 80
cd fig13
# ./run.sh <download dir> <LoRA distribution> <LoRA rank> <CUDA_MPS_ACTIVE_THREAD_PERCENTAGE>
./run.sh $HOME/scratch "uniform" 8 80
./run.sh $HOME/scratch "skewed" 16 80
./run.sh $HOME/scratch "none" 32 80
```
* [__FineInfer__](https://github.com/llm-db/FineInfer)

To benchmark FineInfer in the Conda _basline_ environment,
please use `python python_only_dev.py` before execution, and `python python_only_dev.py --quit-dev` after execution.
These two commands allow users to switch between vanilla vLLM and FineInfer.
NVIDIA MPS has a negligible impact on temporal sharing and is therefore not enabled.
```
conda activate baseline
cd ~
git clone -b vllm-v0.6.3-fineinfer https://github.com/llm-db/llmstation.git vllm-v0.6.3-fineinfer
cd vllm-v0.6.3-fineinfer
python python_only_dev.py
python python_only_dev.py --quit-dev
```
  - For Figures 8 and 9, by varying the request rate and _fineinfer defer_ (i.e., defer in seconds before new prefill), you can get the results in result.txt within the same folder.
  - For Figure 10, you only need to adjust _fineinfer defer_  until the PEFT throughput is the same as points on the X-axis.
  - For Figure 13, you need to adjust LoRA distribution, LoRA rank, and _fineinfer defer_.
```
cd FineInfer
cd fig8a
# ./run.sh <download dir> <request rate> <fineinfer defer>
./run.sh $HOME/scratch 1 0.1
cd fig9a
# ./run.sh <download dir> <request rate> <fineinfer defer>
./run.sh $HOME/scratch 1 0.01
cd fig10
# ./run.sh <download dir> <fineinfer defer>
./run.sh $HOME/scratch 0.5
cd fig13
# ./run.sh <download dir> <LoRA distribution> <LoRA rank> <fineinfer defer>
./run.sh $HOME/scratch "uniform" 8 0.01
```
* __chunked-training__

We implement a precise version of chunked-training on top of touchtune.
You can follow the instructions in the [torchtune-v0.4.0-chunked branch](https://github.com/llm-db/llmstation/tree/torchtune-v0.4.0-chunked) to reproduce Figure 4 (just need to vary _chunk_size_).
Please note that do not mix the environment used in that branch with the environment used in this README.md.

For ease of reproduction, we directly use the vanilla torchtune to run a lightweight/faster but lossy version of chunked-training,
so you can benchmark chunked-training in the Conda _basline_ environment.
```
cd chunked
conda activate baseline
```
  - For Figures 8 and 9, by varying the request rate and _chunk_size_, you can get the results in result.txt within the same folder.
  - For Figure 10, you only need to adjust _chunk_size_ until the PEFT throughput is the same as points on the X-axis.
  - For Figure 13, you need to adjust LoRA distribution, LoRA rank, and _chunk_size_.
```
cd fig8a
# ./run.sh <download dir> <request rate> <chunk size>
./run.sh $HOME/scratch 1 128
cd fig9a
# ./run.sh <download dir> <request rate> <chunk size>
./run.sh $HOME/scratch 1 128
cd fig10
# ./run.sh <download dir> <chunk size>
./run.sh $HOME/scratch 128
cd fig13
# ./run.sh <download dir> <LoRA distribution> <LoRA rank> <chunk size>
./run.sh $HOME/scratch "uniform" 8 128
```
* __LLMStation__

We use the Conda _lms_ environment for benchmarking.
```
cd llmstation
conda activate lms
```
LLMStation offers more flexibility in scheduling by allowing the program or the users to adjust _forward_tasklets_, _forward_wait_,
_backward_tasklets_, and _backward_wait_ to adapt to the workloads.
For ease of reproduction, the script here uses fixed parameters entered by the users but they can be dynamically adjusted during runtime as described in the paper.
In short, _forward_tasklets_ is the number of tasklets performed in forward pass before preemption,  _forward_wait_ is the number of seconds to wait after preemption in forward pass,
_backward_tasklets_ is the number of tasklets performed in backward pass before preemption, and _backward_wait_ is the number of seconds to wait after preemption in backward pass.
We recommend users to make _forward_wait_ or _backward_wait_ larger or smaller according to the P99 TTFT/TPOT you got in result.txt within the same folder.
Then you will find that LLMStation can surpass all baselines while meeting the SLOs.
  - For Figures 8 and 9, by varying the request rate and the parameters mentioned above, you can get the results in result.txt within the same folder.
  - For Figure 10, you only need to adjust the parameters mentioned above until the PEFT throughput is the same as points on the X-axis.
  - For Figure 13, you need to adjust LoRA distribution, LoRA rank, and
```
cd fig8a
# ./run.sh <download dir> <request rate> <forward tasklets> <forward wait> <backward tasklets> <backward wait>
./run.sh $HOME/scratch 1 8 0.005 8 0.005
cd fig9a
# ./run.sh <download dir> <request rate> <forward tasklets> <forward wait> <backward tasklets> <backward wait>
./run.sh $HOME/scratch 1 8 0.005 8 0.005
cd fig10
# ./run.sh <download dir> <forward tasklets> <forward wait> <backward tasklets> <backward wait>
./run.sh $HOME/scratch 8 0.005 8 0.005
cd fig13
# ./run.sh <download dir> <LoRA distribution> <LoRA rank> <forward tasklets> <forward wait> <backward tasklets> <backward wait>
./run.sh $HOME/scratch "uniform" 8 8 0.005 8 0.005
```
  - For Figure 12, in order to make the number of pauses for each forward pass or backward pass greater than the number of layers,
    you may need to adjust the number [here](https://github.com/llm-db/llmstation/blob/pytorch-v2.4.0-lms/torch/csrc/autograd/engine.cpp#L902) and then re-install PyTorch.
    If the number of pauses required for each forward or backward pass is less than the number of layers,
    you just need to use the previous scripts and adjust _forward_tasklets_ and _backward_tasklets_.
```
# ./run.sh <download dir>
./run.sh $HOME/scratch
```

### Other issues
* Sequence length: The sequence length of the inference requests in the artifact is exactly the same as in our experiment.
  However, for ease of reproduction, here we pack/pad the sequence length of the fine-tuned samples to around 512, while our experiments use the original samples.
  Otherwise, we need to ensure that the start and end times of the fine-tuning and serving clients are exactly the same to get the results, which is very inconvenient.
  So be aware that packing/padding may make __vLLM+torchtune__ perform slightly worse, and make __FineInfer__ perform slightly better.
* Other models: Llama-3.1-8B has 32 transformer layers and the backward computation graph for its LoRA fine-tuning has about 4k nodes.
  The current customized [PyTorch branch](https://github.com/llm-db/llmstation/blob/pytorch-v2.4.0-lms/torch/csrc/autograd/engine.cpp#L902) enables __LLMStation__ to pause per layer when fine-tuning LLama-3.1-8B.
  However, if you use a different model with a different number of layers and nodes, you may need to adjust the number [here](https://github.com/llm-db/llmstation/blob/pytorch-v2.4.0-lms/torch/csrc/autograd/engine.cpp#L902) and then re-install PyTorch.
* Device placement: There are two ways to deploy LoRA fine-tuning and serving of Llama-3.1-8B.
  - Half of the GPUs are used for LoRA fine-tuning, and half of the GPUs are used for serving. We use this deployment in Figures 8c, 9c, and 11.
  - Set the degree of tensor parallelism to the same as the number of GPUs. We use this deployment in the rest of the Figures and also the scripts in this branch.
* NCCL: vLLM sometimes cannot find NCCL when using customized PyTorch.
```
export VLLM_NCCL_SO_PATH=$HOME/pytorch-v2.4.0-lms/build/nccl/lib/libnccl.so.2.20.5
```
* Attention backend: vLLM will not install xformers for customized PyTorch.
```
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
```
* Kill the processes that did not exit successfully
```
pkill -9 -f -- '--multiprocessing-fork'
pkill -9 -f -- 'vllm'
# or
pkill -9 -u "$(whoami)"
```
* [Install gcc & g++ 13.3.0 from scratch](https://gcc.gnu.org/wiki/InstallingGCC)
```
wget https://ftp.gnu.org/gnu/gcc/gcc-13.3.0/gcc-13.3.0.tar.gz
tar -xf gcc-13.3.0.tar.gz
./contrib/download_prerequisites
mkdir objdir
cd objdir
export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu
../configure --prefix=$HOME/GCC-13 --enable-languages=c,c++ --disable-multilib
make -j32
make install
export PATH=$HOME/GCC-13/bin:$PATH
export LD_LIBRARY_PATH=$HOME/GCC-13/lib64:$LD_LIBRARY_PATH
```
