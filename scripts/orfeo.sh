#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:V100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --job-name=genesis
#SBATCH --output=genesis_%j.log

cd genesis_tests
conda deactivate
source .venv/bin/activate

echo "Starting training encoder"
PYTHONPATH=src python -m train.train_encoder.train config/train_encoder_config.py \
                --encoder_type=PointNet \
                --max_epochs=25 \
        

exit 0