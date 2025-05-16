#!/bin/bash
# $1 - download dir

if [[ $# -lt 1 ]]; then
    echo "Too few arguments. "
    echo "Usage $0 <download dir>"
    exit
fi

download_dir=$1

tune download meta-llama/Meta-Llama-3.1-8B --output-dir "${download_dir}/Meta-Llama-3.1-8B" \
--ignore-patterns "original/consolidated*"

#tune download meta-llama/Llama-2-13b-hf --output-dir "${download_dir}/Llama-2-13b-hf" \
#--ignore-patterns "original/consolidated*"

#tune download meta-llama/Meta-Llama-3.1-70B --output-dir "${download_dir}/Meta-Llama-3.1-70B" \
#--ignore-patterns "original/consolidated*"
