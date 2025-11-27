#!/bin/bash
#SBATCH --job-name=sarthak-semeval
#SBATCH --output=predict_logs/sarthak-semeval-%j.out
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=32G

source /apps/local/anaconda3/conda_init.sh
conda activate semeval

python -m my_task_b.predict \
	--model_path "codet5_focal_20k_checkpoints" \
	--parquet_path "task_B/test.parquet" \
	--output_path "codet5_attention_pooling_focal_20k.csv" \
	--batch_size 32 \
	--max_length 512