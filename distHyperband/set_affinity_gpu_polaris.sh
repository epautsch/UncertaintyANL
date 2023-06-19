#!/bin/bash
num_gpus=$(nvidia-smi -L | wc -l)
gpu=$((${PMI_LOCAL_RANK} % ${num_gpus}))

if [ "${PMI_RANK}" -eq 0 ]; then
    export CUDA_VISIBLE_DEVICES="-1"
    gpu="None"
else
    export CUDA_VISIBLE_DEVICES=$gpu
fi

echo "RANK= ${PMI_RANK} LOCAL_RANK= ${PMI_LOCAL_RANK} gpu= ${gpu}"
exec "$@"
