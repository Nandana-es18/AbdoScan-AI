import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, 
    Orientationd, Resized, ToTensord
)
from monai.inferers import sliding_window_inference

# Import your custom modules
from amos_model import get_model
from amos_dataset import AmosNormalizeD 

# ==========================================================
# CONFIGURATION & ORGAN MAPPING
# ==========================================================
# 1:Spleen, 2:R.Kidney, 3:L.Kidney, 4:Gallbladder, 5:Pancreas, 6:Liver, 7:Stomach
ORGAN_MAP = {
    1: {"name": "Spleen",      "color": "blue"},
    2: {"name": "R. Kidney",   "color": "green"},
    3: {"name": "L. Kidney",   "color": "cyan"},
    4: {"name": "Gallbladder", "color": "yellow"},
    5: {"name": "Pancreas",    "color": "magenta"},
    6: {"name": "Liver",       "color": "red"},
    7: {"name": "Stomach",     "color": "silver"}
}

# ✅ Proper Device Selection for Mac
DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

MODEL_PATH = "outputs/best_amos_model_new.pth"
TEST_DIR = "./data/imageTs"
OUTPUT_DIR = "outputs"


def visualize_test_set():
    print(f"Using device: {DEVICE}")

    # 1. Check for images
    test_images = sorted(glob.glob(os.path.join(TEST_DIR, "*.nii.gz")))
    if not test_images:
        print(f"❌ Error: No images found in {TEST_DIR}")
        return

    image_path = test_images[0]
    print(f"🚀 Processing: {os.path.basename(image_path)}")

    # 2. Preprocessing Pipeline
    test_transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(1.5, 1.5, 2.0), mode="bilinear"),
        AmosNormalizeD(keys=["image"]),
        Resized(keys=["image"], spatial_size=[96, 96, 96]),
        ToTensord(keys=["image"]),
    ])

    # 3. Load Model
    model = get_model().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # 4. Prepare Data
    data_dict = {"image": image_path, "image_meta": image_path}
    data_dict = test_transforms(data_dict)
    inputs = data_dict["image"].unsqueeze(0).to(DEVICE)

    # 5. Inference
    with torch.no_grad():
        output = sliding_window_inference(
            inputs, 
            (96, 96, 96), 
            2 if DEVICE.type == "mps" else 4,  # ✅ safer for MPS memory
            model
        )

        seg = torch.argmax(output, dim=1).detach().cpu().numpy()[0]
        img = inputs.cpu().numpy()[0, 0]

    # 6. Setup Visualization
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

    # ==========================================================
    # LEGEND
    # ==========================================================
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

    # 7. Save and Show
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, "final_report_test_image.png")
    plt.savefig(save_path, bbox_inches='tight', facecolor='black', dpi=150)
    print(f"✅ Success! Report saved to: {save_path}")
    plt.show()


if __name__ == "__main__":
    visualize_test_set()
