if __name__ == "__main__":
    from transformers import AutoImageProcessor, AutoModel
    import torch
    from transformers import pipeline
    from transformers.image_utils import load_image

    pretrained_model_name = "facebook/dinov3-vitB16-pretrain-lvd1689m"
    processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
    dino = AutoModel.from_pretrained(
        pretrained_model_name, 
        device_map="cuda",
        )
    for param in dino.parameters():
        param.requires_grad = False  # Freeze the feature extractor
    

    from data import ImageRotationDataset
    import numpy as np

    def get_random_image(depth = False):
        dataset = ImageRotationDataset("datasets/data_lungs_20", depth=depth)
        idx = np.random.randint(len(dataset))
        print(f"Selected index: {idx}")
        rgbd, rotation = dataset[idx]

        return rgbd, rotation
    
    from utils.images import show_image
    from torchvision.transforms.functional import pil_to_tensor
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = pil_to_tensor(load_image(url))
    image = image.numpy()

    while True:
        image, rotation = get_random_image(depth = False)

        print(image.numpy(), image.shape)
        print("Minimum pixel ", torch.min(image))
        print("Maximum pixel ", torch.max(image))
        show_image(image.permute(1,2,0).detach().cpu().numpy())
        
        


        inputs = processor(images=image.to("cuda"), return_tensors="pt", 
                                # do_rescale=True,
                                # do_resize=True,
                                # do_center_crop=True,
                                                    )
        
        print(inputs.pixel_values, inputs.pixel_values.shape)
        print("Minimum pixel ", torch.min(inputs.pixel_values))
        print("Maximum pixel ", torch.max(inputs.pixel_values))

        show_image(inputs.pixel_values.squeeze(0).permute(1,2,0).detach().cpu().numpy())

        outputs = dino(**inputs)
        x = outputs.last_hidden_state  # (batch_size, seq_len, feature_dim)

        show_image(x.squeeze(0).detach().cpu().numpy())

        a = input()

