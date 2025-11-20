import torch 
# -----------------------------------------------------------------------------
#entity
entity = "lungs"
# data
dataset = f"data_{entity}_20"
parallel_show = True
feature_analysis = False
# model
model_path = "lightning_logs/genesis-tests/d1lgdxxx/checkpoints/last.ckpt"
dino = "v3" # v2 or v3
model_class = "RotationPredictor" # v2, v3, v3_extractor
epochs = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# -----------------------------------------------------------------------------
