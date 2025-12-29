train_dir = "datasets/lungs_bronchi"
val_dir: Optional[str] = None
test_dir: Optional[str] = None
batch_size = 16
num_workers = 0
img_size = 224
shuffle = True

# Model
n_vertices = 4461  # Number of vertices in the mesh
latent_dim = 32
pretrained_decoder_path = "lightning_logs/train_encoder/mesh_autoencoder/7tx902xs/checkpoints/last.ckpt"  # REQUIRED: Path to pretrained autoencoder checkpoint
pretrained_resnet = True  # Use pretrained ResNet18
freeze_decoder = True  # Freeze decoder weights during training
learning_rate = 1e-4
compile_model = False

# Learning Rate Scheduling and Warmup
use_lr_scheduler = True
scheduler_type = "cosine"  # Options: "cosine", "linear", "exponential", "step"
warmup_epochs = 2
warmup_start_lr = 1e-6
cosine_final_lr = 1e-6  # For cosine scheduler
step_size = 10  # For step scheduler (reduce LR every N epochs)
gamma = 0.1  # For step/exponential scheduler (multiply LR by gamma)

# Trainer
max_epochs = 50
accelerator: Union[str, int] = "auto"
devices: Union[str, List[int]] = "auto"
precision = "bf16-mixed"
default_root_dir = "lightning_logs/train_any"
experiment_name = "image_to_mesh"
limit_train_batches: Union[int, float] = 1.0
limit_val_batches: Union[int, float] = 1.0
limit_test_batches: Union[int, float] = 1.0
checkpoint_name = "image_to_mesh-{epoch:02d}-{val_loss:.4f}"
resume_from: Optional[str] = None

# Logging
use_wandb = True
use_tensorboard_logger = False
wandb_project = "image_to_mesh"
wandb_entity: Optional[str] = None
wandb_group: Optional[str] = None
wandb_tags: List[str] = ["any_dataset", "pretrained_decoder"]
wandb_log_model: Union[str, bool] = "all"
wandb_offline = False

