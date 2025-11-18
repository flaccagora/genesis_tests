from models import DeformNet_v2, DeformNet_v3, DeformNet_v3_extractor
import torch
import numpy as np
import genesis as gs # type: ignore
from data import ImageRotationDataset, show_images
from utils.rotation import rotate_entity

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
    if entity != "dragon":
        scene.add_entity(morph=gs.morphs.Plane())
        pos = (0.5,0.4,0.3)

    E, nu = 3.e4, 0.45
    rho = 1000.


    torus_fem_0 = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=f'assets/{entity_name}.obj',
            pos=pos,
            scale=0.2,
            ),
        material=gs.materials.FEM.Muscle(
            E=E,
            nu=nu,
            rho=rho,
            model='stable-neohooken',
        ),
    )

    torus_fem_1 = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=f'assets/{entity_name}.obj',
            pos=pos,
            scale=0.2,
            ),
        material=gs.materials.FEM.Muscle(
            E=E,
            nu=nu,
            rho=rho,
            model='stable-neohooken',
        ),
    )

    if entity == "dragon":

        cam = scene.add_camera(
            res    = (640, 480),
            pos    = (0,-1,90),
            lookat = (1,1,1),
            fov    = 30,
            GUI    = False,
            far    = 500,
        )
    else:
        cam = scene.add_camera(
            res    = (640, 480),
            pos    = (3., 0.4, 0.3), # (3,,) per torus is enough
            fov    = 30,
            GUI    = False,
        )


    ########################## build ##########################
    scene.build()

    return scene, cam

def get_random_image(dataset):
    idx = np.random.randint(len(dataset.samples))
    print("index: ", idx)
    image, rotation = dataset.samples[np.random.randint(len(dataset.samples))]
    image = np.load(image)
    rotation = torch.load(rotation)

    return image, rotation

def get_predicted_rotation(image, trained_model):
    image_pt = torch.tensor([image], dtype=torch.float16).to("cuda")
    with torch.no_grad():
        predicted_rotation = trained_model(image_pt)
    return predicted_rotation.squeeze(0)

"""REMEMBER TO ALWAYS ROTATE FROM A REFERENCE FRAME POSITION
OTHERWISE THE ROTATION WILL ACCUMULATE ERRORS"""

if __name__ == "__main__":

    # -----------------------------------------------------------------------------
    # data
    dataset = "dataset"
    parallel_show = False
    feature_analysis = True
    # model
    models_dir = "trained_models"
    dino = "v3"
    epochs = 10
    model_class = DeformNet_v3 # DeformNet_v2, DeformNet_v3, DeformNet_v3_extractor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # simul
    entity = "Torus"
    # -----------------------------------------------------------------------------
    config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
    exec(open('utils/configurator.py').read()) # overrides from command line or config file
    config = {k: globals()[k] for k in config_keys} # will be useful for logging
    # ------------------------------F-----------------------------------------------

    assert (dino == "v3" and (model_class == DeformNet_v3_extractor or model_class == DeformNet_v3)) or (dino == "v2" and model_class == DeformNet_v2), f"model class {model_class} incompatible with dino {dino}"
    assert feature_analysis or parallel_show, "choose one among feature_analysis or parallel_show"

    if parallel_show:

        # Model setup
        trained_model = model_class(device)
        trained_model.to(device)
        trained_model.load_state_dict(torch.load(f"{models_dir}/model_{dino}_{epochs}_{dataset}.pth"))

        dataset = ImageRotationDataset("datasets/"+dataset)

        # Simul setup
        scene, cam = gs_simul_setup(entity_name=entity)
        i = 0
        if entity == "dragon":
            i = -1
        torus_fem_0 = scene.entities[1+i]
        torus_fem_1 = scene.entities[2+i]


        while True:
            image, rotation = get_random_image(dataset)
            pred_rotation = get_predicted_rotation(image,trained_model)

            print("rotation ", rotation,"predicted rotation ", pred_rotation)
            rotation = rotation.squeeze(0)

            scene.reset()
            rotate_entity(torus_fem_0,rotation[0], rotation[1], rotation[2])
            rotate_entity(torus_fem_1,pred_rotation[0], pred_rotation[1], pred_rotation[2])
            scene.step()
            show_images(cam.render()[0])

    if feature_analysis:
        def feature_extraction_analysis(image1, image2):

            model = DeformNet_v3(device)
            inputs = model.processor(images=torch.tensor([image1,image2], dtype=torch.float16), return_tensors="pt", do_rescale=False).to(device)
            outputs = model.dino(**inputs)
            x = outputs.last_hidden_state

            l = [image1, image2, x[0].cpu().numpy(),x[0].cpu().numpy()]
            show_images(*l)
            return
        
        print(f"showing pairs of images and image features extracted from dino {dino}")

        dataset = ImageRotationDataset("datasets/"+dataset)

        while True:

            image1, rotation = get_random_image(dataset)
            image2, rotation = get_random_image(dataset)
            feature_extraction_analysis(image1, image2)

        