from __future__ import annotations

from typing import Optional

import torch
import numpy as np
import genesis as gs  # type: ignore

from data import ImageActuationRotationDataset
from models import RGB_ActuationRotationPredictor
from train.train_actu.module import DeformNetLightningModule
from utils.configurator import apply_overrides
from utils.rotation import rotate_entity, rot6d_to_rotmat
from utils.images import show_image, show_images
from torchvision import transforms
import torch.nn.functional as F
from loss.loss import GeodesicLoss




def get_random_image(dataset):
    idx = np.random.randint(len(dataset.samples))
    # get only id divisible by 100
    idx = idx - (idx % 100)
    print("index: ", idx)
    image, actu, rotation = dataset[idx]
    return image, actu, rotation

def get_predictions(image, trained_model, device):
    """Get predicted actuation and rotation from the model.
    
    Args:
        image: Input image tensor
        trained_model: The Lightning module with the trained model
        device: Device to run inference on
    
    Returns:
        Predicted actuation and rotation tensors
    """
    print(image.shape)
    with torch.no_grad():
        image_batch = image.unsqueeze(0).to(device)
        pred_actu, pred_rot = trained_model(image_batch)
    return pred_actu.squeeze(0).cpu(), pred_rot.squeeze(0).cpu()

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
    model_cls: str = "RGB_ActuationRotationPredictor",
    backbone: str = "dinov2_vitb14",
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> DeformNetLightningModule:
    """Load model from Lightning checkpoint, consistent with training scripts.
    
    Args:
        checkpoint_path: Path to the Lightning checkpoint (.ckpt file)
        model_cls: Model variant name
        backbone: Backbone type
        device: Device to load the model on
    
    Returns:
        Loaded DeformNetLightningModule in eval mode
    """
    # Load the model from checkpoint - this restores all hyperparameters and weights
    lightning_module = DeformNetLightningModule.load_from_checkpoint(
        checkpoint_path,
        map_location=device,
        model_cls=model_cls,
        backbone=backbone,
    )
    
    # Move to device and set to evaluation mode
    lightning_module = lightning_module.to(device)
    lightning_module.eval()
    
    return lightning_module

def translate_mpm(entity, translation: np.ndarray):
    """Translate an entity by a given translation vector.
    
    Args:
        entity: The Genesis entity to translate
        translation: A numpy array of shape (3,) representing the translation vector
    """
    current_pos = entity.get_particles_pos()
    
    new_pos = current_pos + torch.tensor(translation, dtype=current_pos.dtype, device=current_pos.device)
    entity.set_particles_pos(new_pos)


def rotate_entity_with_matrix(entity, rotation_matrix, center=None):
    """
    Rotate a rigid entity using a rotation matrix, optionally around a specified center point.
    
    Parameters
    ----------
    entity : genesis.Entity
        The rigid entity to rotate.
    rotation_matrix : array_like, shape (3, 3)
        A 3x3 rotation matrix representing the desired rotation.
        The matrix should be orthogonal (R @ R.T = I) and have determinant +1.
    center : array_like, shape (3,), optional
        The center point to rotate around. If None, rotates around the entity's 
        current position (only orientation changes). Defaults to None.
    
    Notes
    -----
    When `center` is provided, the entity will orbit around that point while also
    rotating its orientation. The transformation is:
        new_pos = center + R @ (old_pos - center)
        new_orientation = R @ old_orientation
    
    Example
    -------
    >>> # Create a rotation matrix for 45 degrees around Z-axis
    >>> angle = np.pi / 4
    >>> R_z = np.array([
    ...     [np.cos(angle), -np.sin(angle), 0],
    ...     [np.sin(angle),  np.cos(angle), 0],
    ...     [0,              0,             1]
    ... ])
    >>> # Rotate around origin
    >>> rotate_entity_with_matrix(bronchi, R_z, center=[0, 0, 0])
    >>> # Rotate around a custom point
    >>> rotate_entity_with_matrix(bronchi, R_z, center=[0.5, 0.5, 0.3])
    """
    # Convert rotation matrix to scipy Rotation object
    rotation = R.from_matrix(rotation_matrix)
    
    # Convert to quaternion (scalar-first format: w, x, y, z)
    quat = rotation.as_quat(scalar_first=True)
    
    if center is not None:
        center = np.asarray(center, dtype=np.float64)
        
        # Get current position
        current_pos = entity.get_pos().cpu().numpy().flatten()
        
        # Translate to origin (relative to center), rotate, translate back
        # new_pos = center + R @ (current_pos - center)
        relative_pos = current_pos - center
        rotated_relative_pos = rotation_matrix @ relative_pos
        new_pos = center + rotated_relative_pos
        
        # Set new position
        entity.set_pos(new_pos)
    
    # Get current orientation and compose with new rotation
    current_quat = entity.get_quat().cpu().numpy().flatten()
    current_rotation = R.from_quat(current_quat, scalar_first=True)
    
    # Compose rotations: new_rotation = rotation * current_rotation
    # This applies the new rotation in the world frame
    new_rotation = rotation * current_rotation
    new_quat = new_rotation.as_quat(scalar_first=True)
    
    # Set the entity's quaternion
    entity.set_quat(new_quat)


def rotate_mpm_entity_with_matrix(entity, rotation_matrix, center=None):
    """
    Rotate an MPM (particle-based) entity using a rotation matrix.
    
    Parameters
    ----------
    entity : genesis.MPMEntity
        The MPM entity to rotate (e.g., lungs with MPM.Elastic material).
    rotation_matrix : array_like, shape (3, 3)
        A 3x3 rotation matrix representing the desired rotation.
        The matrix should be orthogonal (R @ R.T = I) and have determinant +1.
    center : array_like, shape (3,), optional
        The center point to rotate around. If None, rotates around the centroid 
        of all particles. Defaults to None.
    
    Notes
    -----
    This function rotates all particles of the MPM entity around the specified center.
    The transformation applied to each particle position is:
        new_pos = center + R @ (old_pos - center)
    
    Example
    -------
    >>> # Create a rotation matrix for 45 degrees around Z-axis
    >>> angle = np.pi / 4
    >>> R_z = np.array([
    ...     [np.cos(angle), -np.sin(angle), 0],
    ...     [np.sin(angle),  np.cos(angle), 0],
    ...     [0,              0,             1]
    ... ])
    >>> # Rotate around the centroid of particles
    >>> rotate_mpm_entity_with_matrix(lungs, R_z)
    >>> # Rotate around a custom center point
    >>> rotate_mpm_entity_with_matrix(lungs, R_z, center=[0.0, 0.0, 0.4])
    """
    # Get current particle positions
    current_positions = entity.get_particles_pos()  # Shape: (n_particles, 3) or (n_envs, n_particles, 3)
    
    # Handle both single env and multi-env cases
    squeeze_output = current_positions.ndim == 2
    if squeeze_output:
        current_positions = current_positions.unsqueeze(0)  # Add env dimension
    
    # Convert rotation matrix to torch tensor
    rotation_matrix_torch = rotation_matrix.detach().clone().to(current_positions.device)
    # set rotation matrix dtype to current_positions.dtype
    rotation_matrix_torch = rotation_matrix_torch.to(current_positions.dtype)
    # Determine center of rotation
    if center is None:
        # Use centroid of all particles
        center_torch = current_positions.mean(dim=1, keepdim=True)  # Shape: (n_envs, 1, 3)
    else:
        center_torch = torch.tensor(center, dtype=current_positions.dtype, device=current_positions.device)
        center_torch = center_torch.view(1, 1, 3).expand(current_positions.shape[0], 1, 3)
    
    # Apply rotation: new_pos = center + R @ (old_pos - center)
    relative_positions = current_positions - center_torch  # Shape: (n_envs, n_particles, 3)
    
    # Apply rotation matrix to each particle position
    # relative_positions: (n_envs, n_particles, 3)
    # rotation_matrix_torch: (3, 3)
    # Result: (n_envs, n_particles, 3)
    rotated_relative = torch.einsum('ij,...j->...i', rotation_matrix_torch, relative_positions)
    
    new_positions = center_torch + rotated_relative
    
    # Set the new particle positions
    if squeeze_output:
        new_positions = new_positions.squeeze(0)
    
    entity.set_particles_pos(new_positions)
    
    # Also zero out velocities to prevent instabilities after rotation
    zero_vels = torch.zeros_like(new_positions)
    entity.set_particles_vel(zero_vels)


def reset_scene_with_mpm(scene, state):
    """
    Properly reset a scene containing MPM entities, ensuring visual state is updated.
    
    Parameters
    ----------
    scene : genesis.Scene
        The scene to reset.
    state : genesis.SimState
        The state to reset to (typically obtained from scene.get_state() right after build).
    
    Notes
    -----
    For MPM entities, the standard scene.reset() may not properly update the visualization
    because the render fields (particles_render, vverts_render) need to be explicitly 
    synced from the simulation state. This function handles that synchronization.
    """
    # CRITICAL: Clear ALL dynamic nodes BEFORE reset
    # Dynamic nodes are keyed by time step, and after reset scene._t = 0
    # Old nodes from higher time steps won't be auto-cleared
    scene._visualizer._context.clear_dynamic_nodes(only_outdated=False)
    
    # Reset scene with the stored initial state
    scene.reset(state=state)
    
    # Force update the MPM solver's render fields
    # This syncs the visualization particles with the reset simulation state
    if scene.mpm_solver.is_active:
        scene.mpm_solver.update_render_fields()
    
    # Force update the visualizer context with force_render=True
    # This ensures the visual meshes are rebuilt from the reset particle positions
    scene._visualizer._context.update(force_render=True)
    
    # Also force update the viewer if it exists
    if scene._visualizer._viewer is not None:
        scene._visualizer._viewer.update(auto_refresh=True, force=True)



if __name__ == "__main__":

    # -----------------------------------------------------------------------------
    # data
    dataset = "lungs_bronchi"
    img_size = 224
    checkpoint_path = "model_actu_rot.ckpt"  # Path to Lightning checkpoint (.ckpt)
    model_cls = "RGB_ActuationRotationPredictor" 
    backbone = "dinov2_vits14" 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # simul
    show_viewer = True
    # -----------------------------------------------------------------------------
    config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str, type(None)))]
    apply_overrides(globals()) # overrides from command line or config file
    config = {k: globals()[k] for k in config_keys} # will be useful for logging
    # -----------------------------------------------------------------------------

    assert checkpoint_path is not None, "checkpoint_path must be provided"

    # Model setup - consistent with Lightning training
    trained_model = load_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        model_cls=model_cls,
        backbone=backbone,
        device=device,
    )

    criterion_actu = torch.nn.MSELoss()
    criterion_rot = GeodesicLoss()

    # Build transforms consistent with training datamodule
    transform = build_transforms(img_size=img_size)

    dataset = ImageActuationRotationDataset("datasets/"+dataset, transform=transform)

    # Simul setup


    def init_scene(log_level):
        ########################## init ##########################
        gs.init(backend=gs.gpu, precision="32", logging_level=log_level)

        ########################## create a scene ##########################
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=4e-3,
                substeps=10,
                gravity=(0, 0, -9.8),
            ),
            mpm_options=gs.options.MPMOptions(
                lower_bound=(-0.5, -0.5, -0.4),
                upper_bound=(0.5, 0.9, 1.0),
                grid_density=32,
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(1.5, 1.5, 1.0),
                camera_lookat=(0.0, 0.0, 0.3),
                camera_fov=40,
                max_FPS=60,
            ),
            vis_options=gs.options.VisOptions(
                visualize_mpm_boundary=False,
                rendered_envs_idx=[0],
            ),
            show_viewer=True,
        )

        ########################## entities ##########################
        
        # Ground plane
        plane = scene.add_entity(
            morph=gs.morphs.Plane(),
        )

        # lung_surface=gs.surfaces.Rough(
        #     diffuse_texture=gs.textures.ImageTexture(
        #         image_path="assets/textures/all_low_lunghs_BaseColor.lungh_part01.jpeg",
        #     )
        # )

        # bronchi_surface=gs.surfaces.Rough(
        #     diffuse_texture=gs.textures.ImageTexture(
        #         image_path="assets/textures/all_low_lunghs_BaseColor.lungh_part02.jpeg",
        #     )
        # )

        # Lungs - MPM Elastic material (soft deformable tissue)
        lungs = scene.add_entity(
            material=gs.materials.MPM.Muscle(
                E=5e3,       # Young's modulus - relatively soft for lung tissue
                nu=0.4,      # Poisson's ratio
                rho=500.0,   # Density (lung tissue is less dense than water)
            ),
            morph=gs.morphs.Mesh(
                file="assets/lung_lobes.obj",
                pos=(0.0, 0.0, 0.25),
                scale=0.2,
                euler=(0, 0, 0),
            ),
            surface=gs.surfaces.Default(
                color=(0.9, 0.6, 0.6, 0.8),  # Pinkish color for lung tissue
                vis_mode="visual",
            ),
        )

        # Bronchi - Rigid material with coupling enabled
        # needs_coup=True enables coupling with soft materials (MPM)
        bronchi = scene.add_entity(
            material=gs.materials.MPM.Elastic(
                # needs_coup=True,
                # coup_friction=0.5,
                # coup_softness=0.0,
            ),
            morph=gs.morphs.Mesh(
                file="assets/bronchi.obj",
                pos=(0.0, 0.0, 0.2),  # Same position as lungs to be embedded
                scale=0.2,
                euler=(0, 0, 0),
                fixed=False,  # Bronchi can be moved/rotated
            ),
            surface=gs.surfaces.Default(
                color=(0.8, 0.7, 0.6, 1.0),  # Slightly tan/brown color for bronchi
            ),
        )

        lungs_1 = scene.add_entity(
            material=gs.materials.MPM.Muscle(
                E=5e3,       # Young's modulus - relatively soft for lung tissue
                nu=0.4,      # Poisson's ratio
                rho=500.0,   # Density (lung tissue is less dense than water)
            ),
            morph=gs.morphs.Mesh(
                file="assets/lung_lobes.obj",
                pos=(0.0, 0.0, 0.25),
                scale=0.2,
                euler=(0, 0, 0),
            ),
            surface=gs.surfaces.Default(
                color=(0.4, 0.4, 0.4, 0.4),  # Pinkish color for lung tissue
                vis_mode="visual",
            ),
        )

        # Bronchi - Rigid material with coupling enabled
        # needs_coup=True enables coupling with soft materials (MPM)
        bronchi_1 = scene.add_entity(
            material=gs.materials.MPM.Elastic(
                # needs_coup=True,
                # coup_friction=0.5,
                # coup_softness=0.0,
            ),
            morph=gs.morphs.Mesh(
                file="assets/bronchi.obj",
                pos=(0.0, 0.0, 0.2),  # Same position as lungs to be embedded
                scale=0.2,
                euler=(0, 0, 0),
                fixed=False,  # Bronchi can be moved/rotated
            ),
            surface=gs.surfaces.Default(
                color=(0.4, 0.4, 0.4, 0.4),  # Pinkish color for lung tissue
            ),
        )



        ########################## camera ##########################
        cam = scene.add_camera(
            res=(1280, 960),
            pos=(1.5, 1.5, 1.0),
            lookat=(0.0, 0.0, 0.3),
            fov=40,
            GUI=False,
        )

        ########################## build ##########################
        scene.build()
        initial_state = scene.get_state()
        
        return scene, initial_state, lungs, bronchi, cam, lungs_1, bronchi_1

    scene, initial_state, lungs, bronchi, cam, lungs_1, bronchi_1 = init_scene("info")

    ########################## simulation loop ##########################
    
    # Define a center point for rotation (e.g., camera lookat)
    rotation_center = np.array([0.0, 0.0, 0.3])
    
    
    rotate = rotate_mpm_entity_with_matrix

    while True:
        image, actu, rotation = get_random_image(dataset)
        pred_actu, pred_rot_6d = get_predictions(image, trained_model, device)
        
        # Convert 6D rotation representation to 3x3 rotation matrix
        pred_rot_mat = rot6d_to_rotmat(pred_rot_6d.unsqueeze(0)).squeeze(0)

        # loss_actu = criterion_actu(pred_actu, actu)
        # loss_rot = criterion_rot(pred_rot_mat, rotation)

        # print(f"Actuation Loss: {loss_actu.item():.6f}")
        # print(f"Rotation Loss: {loss_rot.item():.6f}")

        print(f"Actuation GT: {actu.item():.4f}, Pred: {pred_actu.item():.4f}")
        print("Rotation GT:\n", rotation)
        print("Rotation Pred:\n", pred_rot_mat)

        
        rotate(lungs, rotation, rotation_center)
        lungs.set_actuation(actu)

        rotate(bronchi, pred_rot_mat, rotation_center)



        rotate(lungs_1, pred_rot_mat, rotation_center)
        lungs_1.set_actuation(pred_actu)

        rotate(bronchi_1, pred_rot_mat, rotation_center)

        # translate_mpm(bronchi_1, np.array([1.0, 0.0, 0.0]))
        # translate_mpm(lungs_1, np.array([1.0, 0.0, 0.0]))

        for i in range(10):
            scene.step()
        
        if show_viewer:
            input("Press Enter to continue...")
        else:
            show_images(cam.render()[0])
        
        reset_scene_with_mpm(scene, initial_state)
