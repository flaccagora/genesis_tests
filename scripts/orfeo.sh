#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
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
                --encoder_type=pointnet \
                --max_epochs=100 \
                --use_tnet=True
        

# PYTHONPATH=src python -m train.train_any.train config/train_any_config.py \
#                         --pretrained_decoder_path = "lightning_logs/train_encoder/mesh_autoencoder/zco1axri/checkpoints/last.ckpt" \
#                         --max_epochs=25 
                        
exit 0