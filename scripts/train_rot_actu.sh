#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:V100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --job-name=rotactu
#SBATCH --output=rotactu_%j.log

conda deactivate
source .venv/bin/activate

PYTHONPATH=src python -m train.train_actu.train config/train_actu_config.py