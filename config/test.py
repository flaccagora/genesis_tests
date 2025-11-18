import torch 
from src.models import DeformNet_v3, DeformNet_v3_extractor  
# -----------------------------------------------------------------------------
#entity
entity = "lungs"
# data
dataset = f"data_{entity}_20"
parallel_show = True
feature_analysis = False
# model
model_path = "lightning_logs/genesis-tests/184pqxxj/checkpoints/last.ckpt"
dino = "v3" # v2 or v3
model_class = "v3" # v2, v3, v3_extractor
epochs = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# -----------------------------------------------------------------------------
