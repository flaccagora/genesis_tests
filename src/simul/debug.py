from __future__ import annotations

import genesis as gs
import torch  # type: ignore
from utils.images import show_image, show_images
from utils.rotation import generate_random_rotation_matrix, rotate_rigid_entity, rotate_entity

import numpy as np

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
        show_viewer=True,
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


    material = gs.materials.MPM.Muscle(
            E=E,
            nu=nu,
            rho=rho,
            model='stable-neohooken',
    )

    # if entity_name != "lungs":
    #     surface = None
    surface=gs.surfaces.Rough(
        diffuse_texture=gs.textures.ImageTexture(
            image_path="assets/textures/all_low_lunghs_BaseColor.lungh_part01.jpeg",
        )
    )

    lung_lobes = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=f'assets/{entity_name}.obj',
            pos=pos,
            scale=0.2,
            ),
        material=material,
        surface = surface
    )

    bronchi = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=f'assets/bronchi.obj',
            pos=pos,
            scale=0.2,
            fixed=True,  # Bronchi is fixed in place

            ),
        material=gs.materials.Rigid(
            needs_coup=True,
            coup_friction=0.5,
            coup_softness=0.0,
        ),

        surface = surface
    )   

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

    return scene, cam, (lung_lobes, bronchi)



def main(entity_name: str = "lungs") -> None:
    scene, cam, (lung_lobes, bronchi) = gs_simul_setup(entity_name)

    import IPython
    IPython.embed()  # noqa: T100

    center = torch.tensor([0.5, 0.4, 0.3])
    R = generate_random_rotation_matrix(1)
    rotate_rigid_entity(bronchi,R.squeeze(0),center=center)
    rotate_entity(lung_lobes,R,center=center)
    scene.step()


    for i in range(10000):
        actu = np.array([0.2 * (0.5 + np.sin(0.01 * np.pi * i))])
        print(f"step {i}, actu: {actu}")

        lung_lobes.set_actuation(actu)
        scene.step()

        # R = generate_random_rotation_matrix()
        # print(R)
        # rotate_entity(entity, R)
        # scene.step()

        a = input()
      
if __name__ == "__main__":
    main("lung_lobes")
