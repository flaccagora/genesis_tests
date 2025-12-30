#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:V100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --job-name=genesis
#SBATCH --output=genesis_%j.log

# Parse command line arguments
OPTION=""
while [[ $# -gt 0 ]]; do
    case $1 in
        -o|--option)
            OPTION="$2"
            shift 2
            ;;
        dataset|encoder|any)
            OPTION="$1"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--option|-o] {dataset|encoder|any}"
            echo "   or: $0 {dataset|encoder|any}"
            exit 1
            ;;
    esac
done

# If no option provided, show usage
if [[ -z "$OPTION" ]]; then
    echo "Error: No option specified"
    echo "Usage: $0 [--option|-o] {dataset|encoder|any}"
    echo "   or: $0 {dataset|encoder|any}"
    echo ""
    echo "Options:"
    echo "  dataset  - Construct dataset"
    echo "  encoder  - Train encoder"
    echo "  any      - Train any"
    exit 1
fi

cd genesis_tests
conda deactivate
source .venv/bin/activate

case "$OPTION" in
    dataset)
        echo "Constructing dataset"
        PYTHONPATH=src python -m simul.lungs_bronchi_scene -d lungs_bronchi_1
        ;;
    encoder)
        echo "Starting training encoder"
        PYTHONPATH=src python -m train.train_encoder.train config/train_encoder_config.py \
                        --encoder_type=pointnet \
                        --use_tnet=True \
        ;;
    any)
        echo "Starting training any"
        PYTHONPATH=src python -m train.train_any.train config/train_any_config.py \
                                --pretrained_decoder_path = "lightning_logs/train_encoder/mesh_autoencoder/zco1axri/checkpoints/last.ckpt" \
        ;;
    *)
        echo "Error: Invalid option '$OPTION'"
        echo "Valid options are: dataset, encoder, any"
        exit 1
        ;;
esac
                        
exit 0



# Script updated with command-line argument support. You can run it in these ways:
# ./orfeo.sh dataset - to construct the dataset
# ./orfeo.sh encoder - to train the encoder
# ./orfeo.sh any - to train any
# ./orfeo.sh --option dataset or ./orfeo.sh -o encoder - using flag syntax
# The script will only execute the selected option and show a usage message if no option is provided.
