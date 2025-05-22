#!/bin/bash
#SBATCH --gpus-per-node=2
#SBATCH --constraint=48GB
#SBATCH --job-name=pdfparse
#SBATCH -o /storage/usmanova/repor_gri/logs/run_parse_%j.out # STDOUT

export CUDA_VISIBLE_DEVICES=0,1
export EASYOCR_MODULE_PATH='/storage/usmanova/.EasyOCR/'

# python ./src/preprocess/pdf2json.py --report 'Suncor Sustainability 2023'
python ./src/preprocess/parse_docling.py --report 'Suncor Sustainability 2023'