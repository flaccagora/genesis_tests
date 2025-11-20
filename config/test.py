import torch 
# -----------------------------------------------------------------------------
#entity
entity = "lungs"
# data
dataset = f"data_{entity}_20"
parallel_show = False
feature_analysis = True
# model
model_path = "lightning_logs/genesis-tests/7qs4mzrt/checkpoints/last.ckpt"
dino = "v3" # v2 or v3
model_class = "v3" # v2, v3, v3_extractor
epochs = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# -----------------------------------------------------------------------------
