from monai.networks.nets import UNet
import torch

def get_model(device):
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=8,  # Background + 7 organs
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
    return model