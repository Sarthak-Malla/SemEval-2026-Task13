#!/bin/bash
#SBATCH --job-name=sarthak-semeval
#SBATCH --output=train_logs/sarthak-semeval-%j.out
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=32G

source /apps/local/anaconda3/conda_init.sh
conda activate semeval

python -m my_task_b.train \
	--task B \
	--output_dir "./codet5_checkpoints" \
	--epochs 10 \
	--eval_steps 500 \
	--save_steps 1000