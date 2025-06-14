#!/bin/bash
#SBATCH --gpus-per-node=1
#SBATCH --constraint=48GB
#SBATCH --job-name=gri
#SBATCH -o /storage/usmanova/report_gri/logs/run_match_%j.out # STDOUT

export CUDA_VISIBLE_DEVICES=0,1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

python ./matching/retrieval.py \
    --report 'paypal-global-2023'

# python ./matching/llm_match.py \
#     --model 'gpt-3.5-turbo-1106' \
#     --report 'meta-sustainability-2023' \
#     --run_type 'zero-shot'