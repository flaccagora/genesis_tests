"""
Configuration overrides for `train/train_actu/train.py`.
Usage:
    PYTHONPATH=src python src/train/train_actu/train.py config/train_actu_config.py
"""

# Data -------------------------------------------------------------------------
train_dir = "datasets/lungs_bronchi"
val_dir = None
test_dir = None
batch_size = 128
num_workers = 8
img_size = 224 
shuffle = True

# Model ------------------------------------------------------------------------
model_cls = "RGB_ActuationRotationPredictor" 
backbone = "dinov2_vits14" # dinov2_vitb14, dinov2_vits14, dinov3, resnet
compile_model = True
pretrained_path = None
actu_weight = 1.0
rot_weight = 1.0
trans_weight = 1.0
p_init_path = "datasets/init_pos.npy"

# Trainer ----------------------------------------------------------------------
max_epochs = 50
accelerator = "auto"
devices = "auto"
precision = "bf16-mixed"
default_root_dir = "lightning_logs"
experiment_name = "actu_rot_net"
limit_train_batches = 1.0
limit_val_batches = 1.0
limit_test_batches = 1.0
checkpoint_name = "actu_rot-{epoch:02d}-{train_loss_epoch:.4f}"
save_top_k = 4
resume_from = None  

# LR ---------------------------------------------------------------------------
criterion = "mse"  # Options: "mse", "geodesic" (for rotation)
learning_rate = 1e-3
use_lr_scheduler = True
scheduler_type = "cosine"  # Options: "cosine", "linear", "exponential", "step"
warmup_epochs = 2
warmup_start_lr = 1e-6
cosine_final_lr = 1e-6  # For cosine scheduler
step_size = 10  # For step scheduler (reduce LR every N epochs)
gamma = 0.1  # For step/exponential scheduler (multiply LR by gamma)

# Logging ----------------------------------------------------------------------
use_wandb = True
use_tensorboard_logger = False
wandb_project = "genesis-tests"
wandb_entity = None
wandb_group = "actuation"
wandb_tags = ["actuation", "rotation", "dino"]
wandb_log_model = "all"
wandb_offline = False
