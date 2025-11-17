import torch 
from models import DeformNet_v3, DeformNet_v3_extractor  
# -----------------------------------------------------------------------------
# data
dataset = "data_dragon_20"
parallel_show = True
feature_analysis = False
# model
dino = "v3" # v2 or v3
epochs = 10
model_class = DeformNet_v3 # DeformNet_v2, DeformNet_v3, DeformNet_v3_extractor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#entity
entity = "dragon"
