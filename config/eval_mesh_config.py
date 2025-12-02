import torch 
# -----------------------------------------------------------------------------
# entity
entity = "lungs"
show_viewer = False
interactive = False
# data
dataset = f"data_{entity}_20"
depth = False
rgb = True
img_size = 224
# model
model_cls = "RGB_RotationPredictor"  # Options: "RGB_RotationPredictor", "RGBD_RotationPredictor", "Dino_RGB_RotationPredictor"
backbone = "dinov2_vits14"  # Options: 'dinov2_vitb14', 'dinov3', 'resnet'
checkpoint_path = "lightning_logs/genesis-tests/dryl3fuc/checkpoints/last.ckpt"  # y0nvhvxv bx9zb6lt dryl3fuc (RGDB) Path to Lightning checkpoint (.ckpt)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
num_samples=100
# -----------------------------------------------------------------------------
