#!/bin/bash
#SBATCH --job-name=baseline-semeval-sarthak
#SBATCH --output=baseline_results/baseline-semeval-sarthak-%j.out
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=32G


# Activate your environment
source /apps/local/anaconda3/conda_init.sh
conda activate semeval

# Run your script
python -m baselines.train \
	--task B \
	--output_dir "./baseline_results" \
	--epochs 3 \
	--batch_size 16 \
	--learning_rate 2e-5 \
	--max_length 128