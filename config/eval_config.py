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
model_cls = "RGB_RotationPredictor"  # Options: "RGB_RotationPredictor", "RGBD_RotationPredictor", "Dino_RGB_RotationPredictor"
backbone = "dinov2_vitb14"  # Options: 'dinov2_vitb14', 'dinov3_vitb14', 'resnet'
checkpoint_path = "lightning_logs/genesis-tests/e4vljsfc/checkpoints/last.ckpt"  # Path to Lightning checkpoint (.ckpt)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# -----------------------------------------------------------------------------
