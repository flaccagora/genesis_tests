import torch
from src.models import DeformNet_v3_extractor, DeformNet_v3
# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'trained_models'
init_from = 'scratch' # 'scratch' or 'not init_from == "scratch"'
# train
epochs = 20
batch_size = 256
# model
dino="v3"
model_cls = DeformNet_v3
# data
dataset = 'data_Torus_5'
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16' NOT IMPLEMENTED   
compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
