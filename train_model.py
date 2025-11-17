from models import DeformNet_v2, DeformNet_v3, DeformNet_v3_extractor
import torch
import torch.nn as nn
from data import create_dataloader

def train(epochs, bs, data_dir, out_dir, dino="v3", pretrained_model=None, compile=False):
    from tqdm import tqdm
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("USING DEVICE ", device)
    dataloader = create_dataloader("datasets/"+data_dir, batch_size=bs)
    
    if dino == "v3":
        model = DeformNet_v3
    elif dino == "v2":
        model = DeformNet_v2
    else:
        raise ValueError


    if pretrained_model==None:
        deformnet = model(device=device)
    else:
        deformnet = pretrained_model
    deformnet.to(device)
    if compile:
        torch.compile(deformnet)

    num_total = sum(p.numel() for p in deformnet.parameters())
    num_trainable = sum(p.numel() for p in deformnet.parameters() if p.requires_grad)

    print(f"Total parameters: {num_total}")
    print(f"Trainable parameters: {num_trainable}")


    optimizer = torch.optim.Adam(deformnet.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    for epoch in range(epochs):

        epoch_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)

        for images, rotation_matrices in epoch_bar:
            optimizer.zero_grad()
            outputs = deformnet(images.to(device))
            loss = criterion(outputs, rotation_matrices.to(device))
            loss.backward()
            optimizer.step()

            # update tqdm postfix
            epoch_bar.set_postfix(loss=loss.item())

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
        if (epoch+1) % 1 == 0:
            torch.save(deformnet.state_dict(), f"{out_dir}/model_{dino}_{epoch+1}_{data_dir}.pth")

    return deformnet

if __name__ == "__main__":
    import os

    # -----------------------------------------------------------------------------
    # I/O
    out_dir = 'out'
    init_from = 'scratch' # 'scratch' or 'not init_from == "scratch"' or 'gpt2*'
    # train
    epochs = 10
    batch_size = 128
    # model
    dino="v3"
    # data
    dataset = 'openwebtext'
    device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    compile = True # use PyTorch 2.0 to compile the model to be faster
    # -----------------------------------------------------------------------------
    config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
    exec(open('utils/configurator.py').read()) # overrides from command line or config file
    config = {k: globals()[k] for k in config_keys} # will be useful for logging
    # -----------------------------------------------------------------------------

    if dino == "v3":
        model = DeformNet_v3
    elif dino == "v2":
        model = DeformNet_v2
    else:
        raise ValueError

    trained_model = None
    if not init_from == "scratch":
        trained_model = model(device)
        trained_model.load_state_dict(torch.load(f"trained_models/model_{dino}_{epochs}_{dataset}.pth"))

    os.makedirs(out_dir, exist_ok=True)

    trained_model = train(epochs=epochs, bs=batch_size,data_dir=dataset,out_dir=out_dir, pretrained_model=trained_model, compile=compile)
    torch.save(trained_model.state_dict(), f"trained_{dino}_{epochs}_8k.pth")
   



