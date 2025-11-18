import torch 
from models import DeformNet_v3, DeformNet_v3_extractor  
# -----------------------------------------------------------------------------
#entity
entity = "lungs"
# data
dataset = f"data_{entity}_20"
parallel_show = False
feature_analysis = False
# model
models_dir = "trained_models"
dino = "v3" # v2 or v3
model_class = DeformNet_v3 # DeformNet_v2, DeformNet_v3, DeformNet_v3_extractor
epochs = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# -----------------------------------------------------------------------------
