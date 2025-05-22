#!/bin/bash
#SBATCH --gpus-per-node=1
#SBATCH --constraint=48GB
#SBATCH --job-name=gri
#SBATCH -o /storage/usmanova/report_gri/logs/run_match_%j.out # STDOUT

export CUDA_VISIBLE_DEVICES=0,1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# sentence-transformers/all-MiniLM-L6-v2, sentence-transformers/msmarco-MiniLM-L-12-v3
# python src/create_embeddings.py \
#     --model sentence-transformers/all-MiniLM-L6-v2 \
#     --output_path /storage/usmanova/reportkg/taxonomies

python ./src/matching/llm_match.py \
    --model 'meta-llama/Llama-3.1-8B-Instruct' \
    --report 'boeing-sustainability-2023' \
    --output_path '/storage/usmanova/reportkg/gri_matches/meta-llama-3.1B/'

# python ./src/matching/gri_match.py \
#     --ranker bm25 \
#     --reranker msmarco \
#     --topN 10
