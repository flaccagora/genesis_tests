import numpy as np
import genesis as gs # type: ignore

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
    scene.add_entity(morph=gs.morphs.Plane())

    E, nu = 3.e4, 0.45
    rho = 1000.


    torus_fem_0 = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=f'assets/{entity_name}.obj',
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
        pos    = (2., 0.4, 0.3), # (3,,) per torus is enough
        lookat = (0.5, 0.4, 0.3),
        fov    = 30,
        GUI    = False,
    )


    ########################## build ##########################
    scene.build()

    return scene, cam

scene, cam = gs_simul_setup("dragon")
torus_fem_0 = scene.entities[1]

# torus_fem_1 = scene.entities[2]
from data import show_image
while True:
    scene.step()
    show_image(cam.render()[0])
