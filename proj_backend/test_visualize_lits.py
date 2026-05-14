import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from monai.inferers import sliding_window_inference
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd, ScaleIntensityRanged, EnsureTyped, ToTensord
from model_lits import get_lits_model

# Config
IMAGE_PATH = "/Users/merinphilip/Desktop/ProjectAMOS/data/imageTs/amos_0469.nii.gz" # Change this for testing
MODEL_PATH = "outputs/best_lits_model.pth"

# Use MPS for MacBook (Apple Silicon), fallback to CPU if not available
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def visualize():
    model = get_lits_model().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    transforms = Compose([
        LoadImaged(keys=["image"]), EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(1.5, 1.5, 1.5), mode="bilinear"),
        ScaleIntensityRanged(keys=["image"], a_min=-200, a_max=250, b_min=0.0, b_max=1.0, clip=True),
        EnsureTyped(keys=["image"]), ToTensord(keys=["image"]),
    ])

    data = transforms({"image": IMAGE_PATH})
    inputs = data["image"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = sliding_window_inference(inputs, (128, 128, 128), 4, model)
        seg = torch.argmax(output, dim=1).detach().cpu().numpy()[0]
        img = inputs.cpu().numpy()[0, 0]

    # Find the slice with the most tumor (class 2)
    slice_idx = np.argmax(np.sum(seg == 2, axis=(0, 1))) if np.max(seg) == 2 else img.shape[2]//2

    plt.figure(figsize=(10, 5), facecolor='black')
    plt.subplot(1, 2, 1); plt.imshow(img[:, :, slice_idx], cmap="gray"); plt.title("CT Scan", color="white")
    
    rgb = np.zeros((*seg[:, :, slice_idx].shape, 3))
    rgb[seg[:, :, slice_idx] == 1] = [0, 1, 0] # Liver = Green
    rgb[seg[:, :, slice_idx] == 2] = [1, 0, 0] # Tumor = Red
    
    plt.subplot(1, 2, 2); plt.imshow(rgb); plt.title("AI: Green=Liver, Red=Tumor", color="white")
    plt.show()

if __name__ == "__main__":
    visualize()