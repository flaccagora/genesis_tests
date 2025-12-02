from __future__ import annotations

import genesis as gs  # type: ignore
from utils.images import show_image, show_images
from utils.rotation import generate_random_rotation_matrix, rotate_rigid_entity, rotate_entity

def build_scene(entity_name: str):
    gs.init(seed=0, precision="32", logging_level="info")

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
        fem_options=gs.options.FEMOptions(dt=dt),
        vis_options=gs.options.VisOptions(show_world_frame=True),
        show_viewer=True,
    )

    scene.add_entity(morph=gs.morphs.Plane())

    scene.add_entity(
        morph=gs.morphs.Mesh(
            file=f"assets/{entity_name}.obj",
            pos=(0.5, 0.4, 0.3),
            scale=0.2,
        ),
        # material=gs.materials.MPM.Muscle(
        #     E=3.0e4,
        #     nu=0.45,
        #     rho=1000.0,
        #     model="neohooken",
        # ),
    )

    cam = scene.add_camera(
        res=(640, 480),
        pos=(2.0, 0.4, 0.3),
        lookat=(0.5, 0.4, 0.3),
        fov=30,
        GUI=False,
    )

    scene.build()
    return scene, cam


def main(entity_name: str = "lungs") -> None:
    scene, cam = build_scene(entity_name)

    import IPython

    entity = scene.entities[1]
    IPython.embed()  # noqa: T100


    while True:
        R = generate_random_rotation_matrix()
        print(R)
        rotate_rigid_entity(entity, R)
        scene.step()

        a = input()
7       
if __name__ == "__main__":
    main("lung_lobes")
