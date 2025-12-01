from __future__ import annotations

from typing import Optional

import torch
import numpy as np
import genesis as gs  # type: ignore

from data import ImageRotationDataset
from models import RGB_RotationPredictor, RGBD_RotationPredictor, Dino_RGB_RotationPredictor
from train import DeformNetLightningModule
from utils.configurator import apply_overrides
from utils.rotation import rotate_entity, rot6d_to_rotmat
from utils.images import show_image, show_images
from torchvision import transforms

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
        scene.add_entity(morph=gs.morphs.Plane())
        pos = (0.5,0.4,0.3)
    if entity_name == "lungs":
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

    pos_1 = np.array(pos)
    pos_1 = pos_1 + np.array([1,0,0])
    entity_2 = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=f'assets/{entity_name}.obj',
            pos=tuple(pos),
            scale=0.2,
            ),
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

    return scene, cam

def get_random_image(dataset):
    idx = np.random.randint(len(dataset.samples))
    print("index: ", idx)
    image, rotation = dataset[idx]
    return image, rotation

def get_predicted_rotation(image, trained_model, device):
    """Get predicted rotation from the model.
    
    Args:
        image: Input image tensor
        trained_model: The Lightning module with the trained model
        device: Device to run inference on
    
    Returns:
        Predicted rotation tensor
    """
    print(image.shape)
    with torch.no_grad():
        image_batch = image.unsqueeze(0).to(device)
        predicted_rotation = trained_model(image_batch)
    return predicted_rotation.squeeze(0).cpu()

"""REMEMBER TO ALWAYS ROTATE FROM A REFERENCE FRAME POSITION
OTHERWISE THE ROTATION WILL ACCUMULATE ERRORS"""


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
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> DeformNetLightningModule:
    """Load model from Lightning checkpoint, consistent with training scripts.
    
    Args:
        checkpoint_path: Path to the Lightning checkpoint (.ckpt file)
        model_variant: Model variant name (e.g., "RGB_RotationPredictor", "RGBD_RotationPredictor", "Dino_RGB_RotationPredictor")
        device: Device to load the model on
    
    Returns:
        Loaded DeformNetLightningModule in eval mode
    """
    # Load the model from checkpoint - this restores all hyperparameters and weights
    lightning_module = DeformNetLightningModule.load_from_checkpoint(
        checkpoint_path,
        map_location=device,
        model_cls=model_cls,  # Override if needed, otherwise uses saved hparams
    )
    
    # Move to device and set to evaluation mode
    lightning_module = lightning_module.to(device)
    lightning_module.eval()
    
    return lightning_module


if __name__ == "__main__":

    # -----------------------------------------------------------------------------
    # data
    dataset = "dataset"
    parallel_show = False
    feature_analysis = True
    img_size = 224
    # model
    checkpoint_path: Optional[str] = None  # Path to Lightning checkpoint (.ckpt)
    model_cls = "RGB_RotationPredictor"  # Options: "RGB_RotationPredictor", "RGBD_RotationPredictor", "Dino_RGB_RotationPredictor"
    backbone = "dinov2_vitb14"  # Options: 'dinov2_vitb14', 'dinov3_vitb14', 'resnet'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    depth = False
    rgb = True
    # simul
    entity = "Torus"
    # -----------------------------------------------------------------------------
    config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str, type(None)))]
    apply_overrides(globals()) # overrides from command line or config file
    config = {k: globals()[k] for k in config_keys} # will be useful for logging
    # -----------------------------------------------------------------------------

    assert feature_analysis or parallel_show, "choose one among feature_analysis or parallel_show"

    if parallel_show:
        assert checkpoint_path is not None, "checkpoint_path must be provided for parallel_show mode"

        # Model setup - consistent with Lightning training
        trained_model = load_model_from_checkpoint(
            checkpoint_path=checkpoint_path,
            model_cls=model_cls,
            device=device,
        )

        from loss.loss import GeodesicLoss, MSELoss
        criterion = MSELoss()

        # Build transforms consistent with training datamodule
        transform = build_transforms(img_size=img_size)

        dataset = ImageRotationDataset("datasets/"+dataset, transform=transform, rgb=rgb, depth=depth)

        # Simul setup
        scene, cam = gs_simul_setup(entity_name=entity)
        i = 0
        if entity == "dragon" or entity == "lungs":
            i = -1
        torus_fem_0 = scene.entities[1+i]
        torus_fem_1 = scene.entities[2+i]

        rotate = rotate_entity
        if entity == "lungs":
            from utils.rotation import rotate_rigid_entity
            rotate = rotate_rigid_entity

        while True:
            image, rotation = get_random_image(dataset)
            pred_rotation = get_predicted_rotation(image, trained_model, device)
            print(f"Loss (datapoint): {criterion(pred_rotation.to(device), rotation.to(device))}")

            # Convert 6D rotation representation to 3x3 rotation matrix
            pred_rotation = rot6d_to_rotmat(pred_rotation.unsqueeze(0)).squeeze(0)

            print("rotation ", rotation,"\npredicted rotation ", pred_rotation)
            rotation = rotation.squeeze(0)

            print(rotation.shape, pred_rotation.shape)

            scene.reset()
            rotate(torus_fem_0,rotation)
            rotate(torus_fem_1,pred_rotation)
            scene.step()
            show_images(cam.render()[0])

    if feature_analysis:
        def feature_extraction_analysis(image1, image2):
            images = torch.cat((image1.unsqueeze(0),image2.unsqueeze(0)))
            
            # Use Dino_RGB_RotationPredictor for feature extraction analysis
            model = Dino_RGB_RotationPredictor()
            model = model.to(device)
            model.eval()
            
            patch_size = model.dino.config.patch_size

            inputs = model.processor(images=images, return_tensors="pt",do_rescale=False).to(device)
            
            batch_size, _, img_height, img_width = inputs.pixel_values.shape
            num_patches_height, num_patches_width = img_height // patch_size, img_width // patch_size
            num_patches_flat = num_patches_height * num_patches_width

            
            outputs = model.dino(**inputs)
            x = outputs.last_hidden_state

            last_hidden_states = outputs.last_hidden_state
            print(last_hidden_states.shape)  # [1, 1 + 4 + 256, 384]
            assert last_hidden_states.shape == (2, 1 + model.dino.config.num_register_tokens + num_patches_flat, model.dino.config.hidden_size)

            cls_token = last_hidden_states[:, 0, :]
            patch_features_flat = last_hidden_states[:, 1 + model.dino.config.num_register_tokens:, :]
            patch_features = patch_features_flat.unflatten(1, (num_patches_height, num_patches_width))

            print(patch_features.shape, patch_features_flat.shape)

            l = [image1.permute(1,2,0), image2.permute(1,2,0),patch_features_flat[0].cpu(),patch_features_flat[1].cpu(), x[0].cpu().numpy(),x[1].cpu().numpy()]
            # l = [image1.permute(1,2,0), image2.permute(1,2,0),inputs.pixel_values[0].permute(1,2,0).cpu(),inputs.pixel_values[1].permute(1,2,0).cpu(), x[0].cpu().numpy(),x[1].cpu().numpy()]
            show_images(*l)
            return
        
        print(f"showing pairs of images and image features extracted from Dino_RGB_RotationPredictor")

        dataset = ImageRotationDataset("datasets/"+dataset)

        while True:

            image1, rotation = get_random_image(dataset)
            image2, rotation = get_random_image(dataset)
            feature_extraction_analysis(image1, image2)