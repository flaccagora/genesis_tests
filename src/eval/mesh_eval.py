"""Mesh evaluation pipeline for comparing vertex distances between 
ground truth and predicted rotations."""

from __future__ import annotations

from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass

import torch
import numpy as np
import genesis as gs  # type: ignore

from data import ImageRotationDataset
from models import RGB_RotationPredictor, RGBD_RotationPredictor, Dino_RGB_RotationPredictor
from train import DeformNetLightningModule
from utils.configurator import apply_overrides
from utils.rotation import rotate_entity, rot6d_to_rotmat, rotation_matrix_xyz
from utils.images import show_image, show_images
from torchvision import transforms
from loss.loss import GeodesicLoss, MSELoss


@dataclass
class MeshEvalMetrics:
    """Container for mesh evaluation metrics."""
    mean_vertex_distance: float
    max_vertex_distance: float
    min_vertex_distance: float
    std_vertex_distance: float
    rmse: float
    mse: float
    geodesic: float
    num_vertices: int
    
    def __repr__(self) -> str:
        return (
            f"MeshEvalMetrics(\n"
            f"  mean_vertex_distance: {self.mean_vertex_distance:.6f}\n"
            f"  max_vertex_distance:  {self.max_vertex_distance:.6f}\n"
            f"  min_vertex_distance:  {self.min_vertex_distance:.6f}\n"
            f"  std_vertex_distance:  {self.std_vertex_distance:.6f}\n"
            f"  rmse:                 {self.rmse:.6f}\n"
            f"  mse:                  {self.mse:.6f}\n"
            f"  geodesic:             {self.geodesic:.6f}\n"
            f"  num_vertices:         {self.num_vertices}\n"
            f")"
        )


def compute_vertex_distances(
    vertices_gt: torch.Tensor, 
    vertices_pred: torch.Tensor
) -> torch.Tensor:
    """Compute per-vertex Euclidean distances between ground truth and predicted vertices.
    
    Args:
        vertices_gt: Ground truth vertex positions (N, 3) or (B, N, 3)
        vertices_pred: Predicted vertex positions (N, 3) or (B, N, 3)
    
    Returns:
        Per-vertex Euclidean distances (N,) or (B, N)
    """
    return torch.norm(vertices_gt - vertices_pred, dim=-1)


def compute_mesh_metrics(
    vertices_gt: torch.Tensor, 
    vertices_pred: torch.Tensor,
    gt_rotation: Optional[torch.Tensor] = None,
    pred_rotation: Optional[torch.Tensor] = None
) -> MeshEvalMetrics:
    """Compute mesh evaluation metrics from vertex positions.
    
    Args:
        vertices_gt: Ground truth vertex positions (N, 3)
        vertices_pred: Predicted vertex positions (N, 3)
    
    Returns:
        MeshEvalMetrics containing various distance statistics
    """
    distances = compute_vertex_distances(vertices_gt, vertices_pred)
    
    return MeshEvalMetrics(
        mean_vertex_distance=distances.mean().item(),
        max_vertex_distance=distances.max().item(),
        min_vertex_distance=distances.min().item(),
        std_vertex_distance=distances.std().item(),
        rmse=torch.sqrt((distances ** 2).mean()).item(),
        geodesic=GeodesicLoss()(pred_rotation.unsqueeze(0),gt_rotation.unsqueeze(0)).item(),
        mse=MSELoss()(pred_rotation.unsqueeze(0), gt_rotation.unsqueeze(0)).item(),
        num_vertices=vertices_gt.shape[0]
    )


def get_rotated_vertices(
    reference_vertices: torch.Tensor,
    rotation_matrix: torch.Tensor,
    center: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Apply rotation to vertices around a center point.
    
    Args:
        reference_vertices: Original vertex positions (N, 3)
        rotation_matrix: 3x3 rotation matrix
        center: Center of rotation. If None, uses centroid of vertices.
    
    Returns:
        Rotated vertex positions (N, 3)
    """
    # Ensure consistent dtype (float32) and device
    device = reference_vertices.device
    reference_vertices = reference_vertices.float()
    rotation_matrix = rotation_matrix.float().to(device)
    
    if center is None:
        center = reference_vertices.mean(dim=0)
    else:
        center = center.float().to(device)
    
    # Center vertices, rotate, then translate back
    vertices_centered = reference_vertices - center
    vertices_rotated = vertices_centered @ rotation_matrix.T + center
    
    return vertices_rotated


def get_entity_vertices(entity) -> torch.Tensor:
    """Extract vertex positions from a Genesis entity.
    
    Args:
        entity: Genesis entity object
    
    Returns:
        Vertex positions as tensor (N, 3)
    """
    state = entity.get_state()
    pos = state.pos
    # Handle batched positions (1, N, 3) -> (N, 3)
    if pos.dim() == 3:
        pos = pos.squeeze(0)
    return pos

def get_rigid_entity_vertices(entity) -> torch.Tensor:
    """Extract vertex positions from a Genesis entity.
    
    Args:
        entity: Genesis entity object
    
    Returns:
        Vertex positions as tensor (N, 3)
    """
    verts = torch.tensor([])
    for part in entity.geoms:
        vert = torch.tensor(part.mesh.verts)
        verts = torch.cat((verts, vert), dim=0)
    return verts



def gs_simul_setup_for_eval(entity_name: str, show_viewer: bool = False):
    """Setup Genesis simulation for mesh evaluation.
    
    Creates a single entity for reference mesh extraction.
    
    Args:
        entity_name: Name of the entity/mesh file
        show_viewer: Whether to show the viewer
    
    Returns:
        Tuple of (scene, camera, entity)
    """
    gs.init(seed=0, precision='32', logging_level='info')

    dt = 5e-4
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            substeps=10,
            gravity=(0, 0, 0),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.5, 0, 0.8),
            camera_lookat=(0.0, 0.0, 0.0),
            camera_fov=40,
        ),
        mpm_options=gs.options.MPMOptions(
            dt=dt,
            lower_bound=(-1.0, -1.0, -0.2),
            upper_bound=(1.0, 1.0, 1.0),
        ),
        fem_options=gs.options.FEMOptions(
            dt=dt,
        ),
        vis_options=gs.options.VisOptions(
            show_world_frame=False,
        ),
        show_viewer=show_viewer,
    )

    # Entity position setup
    pos = (0.5, 1, 0.3)
    if entity_name == "Torus":
        scene.add_entity(morph=gs.morphs.Plane())
        pos = (0.5, 0.4, 0.3)
    if entity_name == "lungs":
        pos = (0.5, 0.4, 0.3)

    E, nu = 3.e4, 0.45
    rho = 1000.

    material = gs.materials.FEM.Muscle(
        E=E,
        nu=nu,
        rho=rho,
        model='stable-neohooken',
    )
    if entity_name == "lungs":
        material = None

    surface = gs.surfaces.Rough(
        diffuse_texture=gs.textures.ImageTexture(
            image_path="assets/textures/all_low_lunghs_BaseColor.lungh_part01.jpeg",
        )
    )
    if entity_name != "lungs":
        surface = None

    entity = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=f'assets/{entity_name}.obj',
            pos=pos,
            scale=0.2,
        ),
        material=material,
        surface=surface
    )

    # Camera setup
    if entity_name == "dragon":
        cam = scene.add_camera(
            res=(640, 480),
            pos=(0, -1, 90),
            lookat=(1, 1, 1),
            fov=30,
            GUI=False,
            far=500,
        )
    elif entity_name == "Torus":
        cam = scene.add_camera(
            res=(640, 480),
            pos=(3., 0.4, 0.3),
            fov=30,
            GUI=False,
        )
    elif "lung" in entity_name:
        cam = scene.add_camera(
            res=(640, 480),
            pos=(2.5, -2, 0.5),
            lookat=(0.5, 0.4, 0.3),
            fov=30,
            GUI=False,
        )
    else:
        raise ValueError(f"Unknown entity: {entity_name}")

    scene.build()

    # Get the entity (skip plane if present)
    entity_idx = 1 if entity_name == "Torus" else 0
    if entity_name == "lungs":
        entity_idx = 0
    
    return scene, cam, scene.entities[entity_idx]


def build_transforms(img_size: int = 224):
    """Build transforms consistent with training datamodule."""
    transform_ops = [
        transforms.ToTensor(),
        transforms.Resize((img_size, img_size)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ]
    return transforms.Compose(transform_ops)


def load_model_from_checkpoint(
    checkpoint_path: str,
    model_cls: str = "RGB_RotationPredictor",
    backbone: str = "dinov2_vitb14",
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> DeformNetLightningModule:
    """Load model from Lightning checkpoint.
    
    Args:
        checkpoint_path: Path to the Lightning checkpoint (.ckpt file)
        model_cls: Model variant name
        backbone: Backbone type
        device: Device to load the model on
    
    Returns:
        Loaded DeformNetLightningModule in eval mode
    """
    lightning_module = DeformNetLightningModule.load_from_checkpoint(
        checkpoint_path,
        map_location=device,
        model_cls=model_cls,
        backbone=backbone,
    )
    
    lightning_module = lightning_module.to(device)
    lightning_module.eval()
    
    return lightning_module


def get_predicted_rotation(
    image: torch.Tensor, 
    trained_model: DeformNetLightningModule, 
    device: torch.device
) -> torch.Tensor:
    """Get predicted rotation from the model.
    
    Args:
        image: Input image tensor
        trained_model: The Lightning module with the trained model
        device: Device to run inference on
    
    Returns:
        Predicted rotation matrix (3, 3)
    """
    with torch.no_grad():
        image_batch = image.unsqueeze(0).to(device)
        predicted_6d = trained_model(image_batch)
        # Convert 6D rotation representation to 3x3 rotation matrix
    return predicted_6d.squeeze(0).cpu()


def evaluate_single_sample(
    reference_vertices: torch.Tensor,
    gt_rotation: torch.Tensor,
    pred_rotation: torch.Tensor,
    center: Optional[torch.Tensor] = None
) -> MeshEvalMetrics:
    """Evaluate mesh vertices for a single sample.
    
    Args:
        reference_vertices: Original vertex positions (N, 3)
        gt_rotation: Ground truth rotation matrix (3, 3)
        pred_rotation: Predicted rotation matrix (3, 3)
        center: Center of rotation
    
    Returns:
        MeshEvalMetrics for this sample
    """
    # Apply rotations to get vertex positions
    vertices_gt = get_rotated_vertices(reference_vertices, gt_rotation, center)
    vertices_pred = get_rotated_vertices(reference_vertices, pred_rotation, center)
    
    return compute_mesh_metrics(vertices_gt, vertices_pred, gt_rotation, pred_rotation)


def run_mesh_evaluation(
    dataset: ImageRotationDataset,
    trained_model: DeformNetLightningModule,
    reference_vertices: torch.Tensor,
    device: torch.device,
    num_samples: Optional[int] = None,
    verbose: bool = True
) -> Tuple[Dict[str, float], List[MeshEvalMetrics]]:
    """Run mesh evaluation over the dataset.
    
    Args:
        dataset: Dataset of images and rotation matrices
        trained_model: Trained model for rotation prediction
        reference_vertices: Reference mesh vertices (N, 3)
        device: Device to run inference on
        num_samples: Number of samples to evaluate (None for all)
        verbose: Whether to print progress
    
    Returns:
        Tuple of (aggregated_metrics_dict, list_of_per_sample_metrics)
    """
    all_metrics: List[MeshEvalMetrics] = []
    all_distances: List[torch.Tensor] = []
    
    n_samples = len(dataset.samples) if num_samples is None else min(num_samples, len(dataset.samples))
    
    center = reference_vertices.mean(dim=0)
    
    for idx in range(n_samples):
        image, gt_rotation = dataset[idx]
        
        # Get prediction
        pred_6d = get_predicted_rotation(image, trained_model, device)
        rotation_matrix = rot6d_to_rotmat(pred_6d)

        
        # Ensure gt_rotation is a 3x3 matrix
        if gt_rotation.dim() == 3:
            gt_rotation = gt_rotation.squeeze(0)
        
        # Compute metrics
        vertices_gt = get_rotated_vertices(reference_vertices, gt_rotation, center)
        vertices_pred = get_rotated_vertices(reference_vertices, rotation_matrix, center)
        
        distances = compute_vertex_distances(vertices_gt, vertices_pred)
        all_distances.append(distances)
        
        metrics = compute_mesh_metrics(vertices_gt, vertices_pred, gt_rotation.to(device), pred_6d.to(device))
        all_metrics.append(metrics)
        
        if verbose and (idx + 1) % 1000 == 0:
            print(f"Evaluated {idx + 1}/{n_samples} samples, "
                  f"current mean dist: {metrics.mean_vertex_distance:.6f}")
    
    # Aggregate metrics
    all_distances_cat = torch.cat(all_distances)
    
    aggregated = {
        "mean_vertex_distance": all_distances_cat.mean().item(),
        "max_vertex_distance": all_distances_cat.max().item(),
        "min_vertex_distance": all_distances_cat.min().item(),
        "std_vertex_distance": all_distances_cat.std().item(),
        "rmse": torch.sqrt((all_distances_cat ** 2).mean()).item(),
        "mse": np.mean([m.mse for m in all_metrics]),
        "geodesic": np.mean([m.geodesic for m in all_metrics]),
        "total_samples": n_samples,
        "total_vertices_evaluated": all_distances_cat.numel(),
        "per_sample_mean_distance_avg": np.mean([m.mean_vertex_distance for m in all_metrics]),
        "per_sample_mean_distance_std": np.std([m.mean_vertex_distance for m in all_metrics]),
    }
    
    return aggregated, all_metrics


def print_evaluation_summary(aggregated_metrics: Dict[str, float]) -> None:
    """Print a summary of the evaluation results."""
    print("\n" + "=" * 60)
    print("MESH EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total samples evaluated:       {aggregated_metrics['total_samples']}")
    print(f"Total vertices evaluated:      {aggregated_metrics['total_vertices_evaluated']}")
    print("-" * 60)
    print("Vertex Distance Statistics (across all samples):")
    print(f"  Mean:                        {aggregated_metrics['mean_vertex_distance']:.6f}")
    print(f"  Std:                         {aggregated_metrics['std_vertex_distance']:.6f}")
    print(f"  Min:                         {aggregated_metrics['min_vertex_distance']:.6f}")
    print(f"  Max:                         {aggregated_metrics['max_vertex_distance']:.6f}")
    print(f"  RMSE:                        {aggregated_metrics['rmse']:.6f}")
    print("-" * 60)
    print("Error Losses:")
    print(f"  MSE:                         {aggregated_metrics['mse']:.6f}")
    print(f"  Geodesic:                    {aggregated_metrics['geodesic']:.6f}")
    print("-" * 60)
    print("Per-Sample Mean Distance:")
    print(f"  Average:                     {aggregated_metrics['per_sample_mean_distance_avg']:.6f}")
    print(f"  Std:                         {aggregated_metrics['per_sample_mean_distance_std']:.6f}")
    print("=" * 60 + "\n")



if __name__ == "__main__":
    # -----------------------------------------------------------------------------
    # Configuration
    # -----------------------------------------------------------------------------
    # data
    dataset = "dataset"
    img_size = 224
    num_samples: Optional[int] = None  # None for all samples
    
    # model
    checkpoint_path: Optional[str] = None  # Path to Lightning checkpoint (.ckpt)
    model_cls = "RGB_RotationPredictor"  # Options: "RGB_RotationPredictor", "RGBD_RotationPredictor", "Dino_RGB_RotationPredictor"
    backbone = "dinov2_vits14"  # Options: 'dinov2_vitb14', 'dinov3_vitb14', 'resnet'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    depth = False
    rgb = True
    
    # simulation
    entity = "lungs"
    show_viewer = False
    
    # evaluation mode
    interactive = True  # If True, shows samples one by one
    # -----------------------------------------------------------------------------
    config_keys = [k for k, v in globals().items() 
                   if not k.startswith('_') and isinstance(v, (int, float, bool, str, type(None)))]
    apply_overrides(globals())
    config = {k: globals()[k] for k in config_keys}
    # -----------------------------------------------------------------------------

    assert checkpoint_path is not None, "checkpoint_path must be provided"

    print(f"Loading model from: {checkpoint_path}")
    print(f"Entity: {entity}")
    print(f"Device: {device}")
    print(f"Dataset: {dataset}")

    # Load model
    trained_model = load_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        model_cls=model_cls,
        backbone=backbone,
        device=device,
    )

    # Build transforms
    transform = build_transforms(img_size=img_size)

    # Load dataset
    dataset = ImageRotationDataset(
        "datasets/"+dataset, 
        transform=transform, 
        rgb=rgb, 
        depth=depth
    )
    print(f"Dataset size: {len(dataset.samples)} samples")

    # Setup simulation and get reference mesh
    print("Setting up simulation...")
    scene, cam, entity_obj = gs_simul_setup_for_eval(entity_name=entity, show_viewer=show_viewer)
    
    # Get reference vertices from the entity
    reference_vertices = get_rigid_entity_vertices(entity_obj)
    print(f"Reference mesh has {reference_vertices.shape[0]} vertices")

    if interactive:
        # Interactive mode: evaluate and show samples one by one
        from loss.loss import GeodesicLoss, MSELoss
        criterion = MSELoss()
        
        while True:
            idx = np.random.randint(len(dataset.samples))
            print(f"\n--- Sample {idx} ---")
            
            image, gt_rotation = dataset[idx]
            pred_rotation = get_predicted_rotation(image, trained_model, device)
            
            if gt_rotation.dim() == 3:
                gt_rotation = gt_rotation.squeeze(0)
            
            # Compute mesh metrics
            metrics = evaluate_single_sample(
                reference_vertices=reference_vertices,
                gt_rotation=gt_rotation.to(device),
                pred_rotation=pred_rotation.to(device),
            )
            print(metrics)
            
            # Rotate entity for visualization
            rotate = rotate_entity
            if entity == "lungs":
                from utils.rotation import rotate_rigid_entity
                rotate = rotate_rigid_entity
            
            scene.reset()
            rotate(entity_obj, gt_rotation)
            scene.step()
            
            if show_viewer:
                input("Press Enter to continue...")
            else:
                rendered = cam.render()[0]
                show_images(rendered)
    else:
        # Batch evaluation mode
        print("\nRunning batch evaluation...")
        aggregated_metrics, per_sample_metrics = run_mesh_evaluation(
            dataset=dataset,
            trained_model=trained_model,
            reference_vertices=reference_vertices,
            device=device,
            num_samples=num_samples,
            verbose=True
        )
        
        print_evaluation_summary(aggregated_metrics)
        
        # Optionally save results
        import json
        results_path = f"mesh_eval_results_{entity}_{dataset}_{checkpoint_path}.json"
        with open(results_path, 'w') as f:
            json.dump(aggregated_metrics, f, indent=2)
        print(f"Results saved to: {results_path}")
