import numpy as np
import genesis as gs
import torch

def gs_simul_setup(entity):
    ########################## init ##########################
    gs.init(seed=0, precision='32', logging_level='warning')

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
    scene.add_entity(morph=gs.morphs.Plane())

    E, nu = 3.e4, 0.45
    rho = 1000.


    entity = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=f'assets/{entity}.obj',
            pos=(0.5, 0.4, 0.3),
            scale=0.2,
            ),
        material=gs.materials.FEM.Muscle(
            E=E,
            nu=nu,
            rho=rho,
            model='stable-neohooken',
        ),
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


    cam = scene.add_camera(
        res    = (640, 480),
        pos    = (3., 0.4, 0.3),
        lookat = (0.5, 0.4, 0.3),
        fov    = 30,
        GUI    = False,
    )


    ########################## build ##########################
    scene.build()

    return scene, cam

def rotation_matrix_xyz(rx, ry, rz):
    Rx = torch.tensor([[1, 0, 0],
                   [0, torch.cos(rx), -torch.sin(rx)],
                   [0, torch.sin(rx), torch.cos(rx)]], dtype=torch.float32)
    
    Ry = torch.tensor([[torch.cos(ry), 0, torch.sin(ry)],
                   [0, 1, 0],
                   [-torch.sin(ry), 0, torch.cos(ry)]], dtype=torch.float32)
    
    Rz = torch.tensor([[torch.cos(rz), -torch.sin(rz), 0],
                   [torch.sin(rz), torch.cos(rz), 0],
                   [0, 0, 1]], dtype=torch.float32)
    
    R = Rz @ Ry @ Rx
    return R

def rotate_entity(entity, rx, ry=None, rz=None, center=None):
    if ry == None or rz == None:
        R = rx
    else:
        R = rotation_matrix_xyz(rx, ry, rz)
    state = entity.get_state()
    pos = state.pos
    if center is not None:
        com = center
    else:   
        com = pos.mean(dim=1)
    pos_centered = pos - com
    pos_rotated = pos_centered @ R.T + com
    entity.set_position(pos_rotated.sceneless())


"""REMEMBER TO ALWAYS ROTATE FROM A REFERENCE FRAME POSITION
OTHERWISE THE ROTATION WILL ACCUMULATE ERRORS"""



if __name__ == "__main__":
    from tqdm import tqdm
    import os
    # -----------------------------------------------------------------------------
    # data
    dataset_name = 'openwebtext'
    n = 20
    entity_name = "Torus"
    # -----------------------------------------------------------------------------
    config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
    exec(open('configurator.py').read()) # overrides from command line or config file
    config = {k: globals()[k] for k in config_keys} # will be useful for logging
    # -----------------------------------------------------------------------------

    os.makedirs(f"datasets/{dataset_name}_{entity_name}_{n}",exist_ok=True)

    scene, cam = gs_simul_setup(entity_name)
    scene_entity = scene.entities[1]

    progress_bar = tqdm(n**3)
    for f1 in range(n):
        for f2 in range(n):
            for f3 in range(n):
                angle = torch.tensor([torch.pi * f1 / n, torch.pi * f2 / n, torch.pi * f3 / n])
                scene.reset()
                rotate_entity(scene_entity, angle[0], angle[1], angle[2], center=None)
                scene.step()
                # save image and rotation matrix
                img = np.array(cam.render()[0])
                # R = rotation_matrix_xyz(angle[0], angle[1], angle[2])
                R = torch.tensor([angle[0], angle[1], angle[2]])
                # save img and R to disk
                img_filename = f"datasets/{dataset_name}_{entity_name}_{n}/image_f1_{f1}_f2_{f2}_f3_{f3}_n_{n}.npy"
                R_filename = f"datasets/{dataset_name}_{entity_name}_{n}/rotation_f1_{f1}_f2_{f2}_f3_{f3}_n_{n}.th"
                np.save(img_filename, img)
                torch.save(R, R_filename)
            progress_bar.update(n)