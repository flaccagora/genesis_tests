import numpy as np
import genesis as gs  # type: ignore
import torch

from utils.configurator import apply_overrides
from utils.rotation import rotate_entity, rotate_MPM_entity

def gs_simul_setup(entity_name):
    ########################## init ##########################
    gs.init(seed=0, precision='32', logging_level='info')

    dt = 5e-4
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            substeps=10,
            gravity=(0, 0, 0),
        ),
        viewer_options= gs.options.ViewerOptions(
            camera_pos=(1.5, 0, 0.8),
            camera_lookat=(0.0, 0.0, 0.0),
            camera_fov=40,
        ),
        mpm_options=gs.options.MPMOptions(
            dt=dt,
            lower_bound=(-1.0, -1.0, -0.2),
            upper_bound=( 1.0,  1.0,  1.0),
        ),
        fem_options=gs.options.FEMOptions(
            dt=dt,
        ),
        vis_options=gs.options.VisOptions(
            show_world_frame=False,
        ),
        show_viewer=False,
    )

    ########################## entities ##########################
    pos=(0.5, 1, 0.3)
    if entity_name == "Torus":
        print("set pos for Torus")
        scene.add_entity(morph=gs.morphs.Plane())
        pos = (0.5,0.4,0.3)
    if "lung" in entity_name    :
        print("set pos for lungs")
        pos=(0.5, 0.4, 0.3)
   
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

    surface=gs.surfaces.Rough(
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
        surface = surface
    )

    # torus_fem_1 = scene.add_entity(
    #     morph=gs.morphs.Mesh(
    #         file='assets/Torus.obj',
    #         pos=(0.5, 0.4, 0.3),
    #         scale=0.2,
    #         ),
    #     material=gs.materials.FEM.Muscle(
    #         E=E,
    #         nu=nu,
    #         rho=rho,
    #         model='stable-neohooken',
    #     ),
    # )

    if entity_name == "dragon":
        cam = scene.add_camera(
            res    = (640, 480),
            pos    = (0,-1,90),
            lookat = (1,1,1),
            fov    = 30,
            GUI    = False,
            far    = 500,
        )
    elif entity_name == "Torus":
        cam = scene.add_camera(
            res    = (640, 480),
            pos    = (3., 0.4, 0.3), # (3,,) per torus is enough
            fov    = 30,
            GUI    = False,
        )
    elif "lung" in entity_name:
        cam = scene.add_camera(
            res    = (640, 480),
            pos    = (2.5,-2,0.5),
            lookat = (0.5, 0.4, 0.3),
            fov    = 30,
            GUI    = False,
        )
    else:
        raise ValueError

    ########################## build ##########################
    scene.build()

    return scene, cam, entity

if __name__ == "__main__":
    from tqdm import tqdm
    import os
    from utils.rotation import generate_random_rotation_matrix
    # -----------------------------------------------------------------------------
    # data
    dataset_name = 'data'
    n = 20
    entity_name ="lungs"
    # -----------------------------------------------------------------------------
    config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
    apply_overrides(globals())  # overrides from command line or config file
    config = {k: globals()[k] for k in config_keys} # will be useful for logging
    # -----------------------------------------------------------------------------

    os.makedirs(f"datasets/{dataset_name}_{entity_name}_{n}",exist_ok=True)

    scene, cam, scene_entity = gs_simul_setup(entity_name)

    rotate = rotate_entity
    if entity_name == "lungs":
        from utils.rotation import rotate_rigid_entity
        rotate = rotate_rigid_entity

    progress_bar = tqdm(n**3)
    for f1 in range(n):
        for f2 in range(n):
            for f3 in range(n):
                # angle = torch.tensor([torch.pi * f1 / n, torch.pi * f2 / n, torch.pi * f3 / n])
                rotation_matrix = generate_random_rotation_matrix(1).squeeze(0)
                scene.reset()
                
                rotate(scene_entity, rotation_matrix, center=None)
                scene.step()
                
                # save image and rotation matrix
                rgb, depth, _, _ = cam.render(rgb=True, depth=True)
                # R = rotation_matrix_xyz(angle[0], angle[1], angle[2])
                # R = torch.tensor([angle[0], angle[1], angle[2]])
                
                # save img and R to disk
                np.save(f"datasets/{dataset_name}_{entity_name}_{n}/rgb_f1_{f1}_f2_{f2}_f3_{f3}_n_{n}.npy", rgb)
                np.save(f"datasets/{dataset_name}_{entity_name}_{n}/depth_f1_{f1}_f2_{f2}_f3_{f3}_n_{n}.npy", depth)
                torch.save(rotation_matrix, f"datasets/{dataset_name}_{entity_name}_{n}/rotation_f1_{f1}_f2_{f2}_f3_{f3}_n_{n}.th")
            progress_bar.update(n)

    # import IPython
    # IPython.embed()