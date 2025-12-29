train_dir = "datasets/lungs_bronchi/particles"
val_dir: Optional[str] = None
test_dir: Optional[str] = None
batch_size = 128
num_workers = 0
shuffle = True

# Model
n_vertices = 4461  # Number of vertices in the mesh
latent_dim = 32
hidden_dims: Optional[List[int]] = None  # Will use defaults from MeshAutoencoder
encoder_type = "mlp"  # Options: "mlp", "pointnet"
use_tnet = False  # Whether to use T-Net (only for PointNet encoder)
learning_rate = 1e-3
compile_model = False
pretrained_path: Optional[str] = None

# Learning Rate Scheduling and Warmup
use_lr_scheduler = True
scheduler_type = "cosine"  # Options: "cosine", "linear", "exponential", "step"
warmup_epochs = 2
warmup_start_lr = 1e-6
cosine_final_lr = 1e-6  # For cosine scheduler
step_size = 10  # For step scheduler (reduce LR every N epochs)
gamma = 0.1  # For step/exponential scheduler (multiply LR by gamma)

# Trainer
max_epochs = 100
accelerator: Union[str, int] = "auto"
devices: Union[str, List[int]] = "auto"
precision = "bf16-mixed"
default_root_dir = "lightning_logs/train_encoder"
experiment_name = "mesh_autoencoder"
limit_train_batches: Union[int, float] = 1.0
limit_val_batches: Union[int, float] = 1.0
limit_test_batches: Union[int, float] = 1.0
checkpoint_name = "mesh_autoencoder-{epoch:02d}-{val_loss:.4f}"
resume_from: Optional[str] = None

# Logging
use_wandb = True
use_tensorboard_logger = False
wandb_project = "mesh_autoencoder"
wandb_entity: Optional[str] = None
wandb_group: Optional[str] = None
wandb_tags: List[str] = []
wandb_log_model: Union[str, bool] = "all"
wandb_offline = False
