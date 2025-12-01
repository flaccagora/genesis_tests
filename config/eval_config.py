import torch 
# -----------------------------------------------------------------------------
# entity
entity = "lungs"
show_viewer = False
# data
dataset = f"data_{entity}_5"
parallel_show = True
feature_analysis = False
depth = False
rgb = True
img_size = 224
# model
model_cls = "RGB_RotationPredictor"  # Options: "RGB_RotationPredictor", "RGBD_RotationPredictor", "Dino_RGB_RotationPredictor"
backbone = "dinov2_vits14"  # Options: 'dinov2_vitb14', 'dinov3', 'resnet'
checkpoint_path = "lightning_logs/genesis-tests/y0nvhvxv/checkpoints/last.ckpt"  # Path to Lightning checkpoint (.ckpt)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# -----------------------------------------------------------------------------
