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
	--output_dir "./codeberta_attention_checkpoints" \
	--wandb_run_name "codeberta_attention" \
	--epochs 5 \
	--max_length 512 \
	--batch_size 32 \
	--gradient_accumulation_steps 1 \
	--eval_steps 500 \
	--save_steps 1000 \
	--pooling "attention"

# This uses 220m
# python -m my_task_b.train-focal \
# 	--task B \
# 	--output_dir "./codet5_focal_20k_checkpoints" \
# 	--epochs 5 \
# 	--max_length 512 \
# 	--batch_size 32 \
# 	--gradient_accumulation_steps 2 \
# 	--max_samples_per_class 20000 \
# 	--eval_steps 500 \
# 	--save_steps 1000 \
# 	--pooling "attention" \
# 	--no_class_weights \
# 	--use_focal_loss