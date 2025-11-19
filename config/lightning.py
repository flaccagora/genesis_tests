"""
Example configuration overrides for `train/train.py`.
Usage:
    PYTHONPATH=src python -m train.train config/lightning.py
"""

# Data -------------------------------------------------------------------------
train_dir = "datasets/data_lungs_20"
val_dir = None
test_dir = None
batch_size = 128
num_workers = 5
img_size = None # if set images are cropped to square of l img_size
shuffle = True
rgb = True
depth = False

# Model ------------------------------------------------------------------------
model_variant = "v3" # v2 v3_ extractor RGBD
compile_model = True
pretrained_path = None

# Trainer ----------------------------------------------------------------------
max_epochs = 25
accelerator = "auto"
devices = "auto"
precision = "bf16-mixed"
default_root_dir = "lightning_logs"
experiment_name = "deformnet_lightning"
limit_train_batches = 1.0
limit_val_batches = 1.0
limit_test_batches = 1.0
checkpoint_name = "deformnet-{epoch:02d}-{val_loss:.4f}"
resume_from = None # "lightning_logs/genesis-tests/184pqxxj/checkpoints/last.ckpt"

# LR ---------------------------------------------------------------------------
learning_rate = 5e-3
use_lr_scheduler = True
scheduler_type = "cosine"  # Options: "cosine", "linear", "exponential", "step"
warmup_epochs = 1
warmup_start_lr = 1e-6
cosine_final_lr = 1e-6  # For cosine scheduler
step_size = 10  # For step scheduler (reduce LR every N epochs)
gamma = 0.05  # For step/exponential scheduler (multiply LR by gamma)

# Logging ----------------------------------------------------------------------
use_wandb = True
use_tensorboard_logger = True
wandb_project = "genesis-tests"
wandb_entity = None
wandb_group = "baseline"
wandb_tags = ["lightning", "deformnet"]
wandb_log_model = "all"
wandb_offline = False

