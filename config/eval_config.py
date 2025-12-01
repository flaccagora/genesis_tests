import torch 
# -----------------------------------------------------------------------------
# entity
entity = "lungs"
# data
dataset = f"data_{entity}_20"
parallel_show = True
feature_analysis = False
depth = False
rgb = True
img_size = 224
# model
checkpoint_path = "lightning_logs/genesis-tests/s3t63e2l/checkpoints/last.ckpt"  # Path to Lightning checkpoint (.ckpt)
model_variant = "RGB_RotationPredictor"  # Options: "RGB_RotationPredictor", "RGBD_RotationPredictor", "Dino_RGB_RotationPredictor"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# -----------------------------------------------------------------------------
