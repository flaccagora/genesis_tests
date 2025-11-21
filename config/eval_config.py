import torch 
# -----------------------------------------------------------------------------
#entity
entity = "lungs"
# data
dataset = f"data_{entity}_20"
parallel_show = True
feature_analysis = False
depth = False
# model
model_path = "lightning_logs/genesis-tests/s3t63e2l/checkpoints/last.ckpt"
model_class = "RGB_RotationPredictor" # v2, v3, v3_extractor
epochs = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# -----------------------------------------------------------------------------
