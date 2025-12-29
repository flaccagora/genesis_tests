"""
Scene demonstrating lungs (MPM elastic material) coupled with bronchi (rigid body).
The bronchi is rigid and interacts with the soft lung tissue through coupling.
"""

import argparse
import os

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

import genesis as gs
from utils.rotation import generate_random_rotation_matrix


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
    rotation_matrix_torch = torch.tensor(rotation_matrix, dtype=current_positions.dtype, device=current_positions.device)
    
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False, help="Show visualization")
    parser.add_argument("-c", "--cpu", action="store_true", default=False, help="Use CPU backend")
    parser.add_argument("-o", "--output", type=str, default="lungs_render.png", help="Output image filename")
    args = parser.parse_args()

    def init_scene():
        ########################## init ##########################
        gs.init(backend=gs.cpu if args.cpu else gs.gpu, precision="32", logging_level="info")

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
                visualize_mpm_boundary=True,
                rendered_envs_idx=[0],
            ),
            show_viewer=args.vis,
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
        
        return scene, initial_state, lungs, bronchi, cam

    # IMPORTANT: Get initial state BEFORE any simulation steps
    # This captures the exact initial configuration of all particles
    scene, initial_state, lungs, bronchi, cam = init_scene()

    ########################## simulation loop ##########################
    
    # Define a center point for rotation (e.g., camera lookat)
    rotation_center = np.array([0.0, 0.0, 0.3])
    
    # Step 1: Run a few initial steps to let the scene settle
    print("Running initial settling steps...")
    for i in range(50):
        scene.step()



    # create folder structure
    # in datasets/lungs_bronchi I wand a folder for every category: RGB depth normal particles rotation actu
    os.makedirs("datasets/lungs_bronchi/RGB", exist_ok=True)
    os.makedirs("datasets/lungs_bronchi/depth", exist_ok=True)
    os.makedirs("datasets/lungs_bronchi/normal", exist_ok=True)
    os.makedirs("datasets/lungs_bronchi/particles", exist_ok=True)
    os.makedirs("datasets/lungs_bronchi/rotation", exist_ok=True)
    os.makedirs("datasets/lungs_bronchi/actu", exist_ok=True)
    
            
    for i in range(10):
        if i % 100 == 0:
            reset_scene_with_mpm(scene, initial_state)
            rotation_matrix = generate_random_rotation_matrix(1).squeeze(0)
            rotate_mpm_entity_with_matrix(lungs, rotation_matrix, center=rotation_center)
            rotate_mpm_entity_with_matrix(bronchi, rotation_matrix, center=rotation_center)

        actu = np.array([0.5 * (0.5 + np.sin(0.01 * np.pi * i))])
        lungs.set_actuation(actu)
        
        scene.step()

        rgb, depth, _, _ = cam.render(rgb=True, depth=True)
        
        # print types
        print("particles: ", lungs.get_particles_pos().shape)

        # save rgb, depth, normal, particles, rotation, actu
        np.save(f"datasets/lungs_bronchi/RGB/{i}.npy", rgb)
        np.save(f"datasets/lungs_bronchi/depth/{i}.npy", depth)
        np.save(f"datasets/lungs_bronchi/particles/{i}.npy", lungs.get_particles_pos().detach().cpu().numpy())
        np.save(f"datasets/lungs_bronchi/rotation/{i}.npy", rotation_matrix.detach().cpu().numpy())
        np.save(f"datasets/lungs_bronchi/actu/{i}.npy", actu)
        

    print("Done!")

    # dummy save and load to check for consistency
    points = lungs.get_particles_pos()
    print(points.shape)


if __name__ == "__main__":
    main()
