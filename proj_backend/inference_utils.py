import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg') # Prevent GUI issues on server
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, 
    Spacingd, Resized, ToTensord
)
from monai.inferers import sliding_window_inference
from amos_model import get_model
from amos_dataset import AmosNormalizeD

# Setup Device explicitly like amos_test_visualize.py
device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

ORGAN_MAP = {
    1: {"name": "Spleen",      "color": "blue"},
    2: {"name": "R. Kidney",   "color": "green"},
    3: {"name": "L. Kidney",   "color": "cyan"},
    4: {"name": "Gallbladder", "color": "yellow"},
    5: {"name": "Pancreas",    "color": "magenta"},
    6: {"name": "Liver",       "color": "red"},
    7: {"name": "Stomach",     "color": "silver"}
}

def load_inference_model(model_path="/Users/merinphilip/Desktop/Main_project/proj_backend/outputs/best_amos_model_new.pth"):
    model = get_model().to(device)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def process_and_visualize(model, image_path, modality, output_dir="./test_results"):
    os.makedirs(output_dir, exist_ok=True)
    
    # Preprocessing Transforms (Inference Mode)
    test_transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(1.5, 1.5, 2.0), mode="bilinear"),
        AmosNormalizeD(keys=["image"]),
        Resized(keys=["image"], spatial_size=[96, 96, 96]),
        ToTensord(keys=["image"]),
    ])

    data_dict = {
        "image": image_path, 
        "image_meta": image_path,
        "modality": modality
    }
    
    data_processed = test_transforms(data_dict)
    input_tensor = data_processed["image"].unsqueeze(0).to(device)

    # Run Inference
    with torch.no_grad():
        output = sliding_window_inference(
            input_tensor, 
            (96, 96, 96), 
            2 if device.type == "mps" else 4,
            model
        )
        seg = torch.argmax(output, dim=1).detach().cpu().numpy()[0]
        img = input_tensor.cpu().numpy()[0, 0]

    # Visualization exactly like amos_test_visualize.py
    depth = img.shape[2]
    slices = [int(depth * 0.35), int(depth * 0.5), int(depth * 0.65)]
    
    fig, axes = plt.subplots(3, 2, figsize=(12, 14))
    fig.patch.set_facecolor('black')

    for i, s_idx in enumerate(slices):
        # Grayscale Scan
        axes[i, 0].imshow(img[:, :, s_idx], cmap="gray")
        axes[i, 0].set_title(f"Original Scan (Slice {s_idx})", color="white")
        axes[i, 0].axis("off")

        # Color Segmentation
        seg_slice = seg[:, :, s_idx]
        rgb_seg = np.zeros((seg_slice.shape[0], seg_slice.shape[1], 3))

        for val, info in ORGAN_MAP.items():
            rgb_seg[seg_slice == val] = plt.cm.colors.to_rgb(info["color"])

        axes[i, 1].imshow(rgb_seg)
        axes[i, 1].set_title(f"AI Prediction (Slice {s_idx})", color="white")
        axes[i, 1].axis("off")

    # Legend
    legend_elements = [
        mpatches.Patch(color=info["color"], label=f"Class {val}: {info['name']}")
        for val, info in ORGAN_MAP.items()
    ]

    fig.legend(
        handles=legend_elements,
        loc='center right',
        bbox_to_anchor=(1.2, 0.5),
        fontsize=12,
        facecolor='white',
        edgecolor='black',
        title="Organ Color Key",
        title_fontsize='13'
    )

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    
    filename = os.path.basename(image_path)
    save_path = os.path.join(output_dir, f"result_{filename}.png")
    plt.savefig(save_path, bbox_inches='tight', facecolor='black', dpi=150)
    plt.close(fig)
    
    return save_path
