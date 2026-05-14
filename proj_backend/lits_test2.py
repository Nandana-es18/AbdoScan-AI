import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, 
    Orientationd, ScaleIntensityRanged, NormalizeIntensityd, 
    EnsureTyped, ToTensord
)
from model_lits import get_lits_model

# ==========================================================
# CONFIGURATION
# ==========================================================
# Example CT: amos_0469.nii.gz | Example MRI: amos_0505.nii.gz
IMAGE_PATH = "/Users/merinphilip/Desktop/ProjectAMOS/data/imageTs/amos_0525.nii.gz"
MODEL_PATH = "outputs/best_lits_model.pth"

# Use MPS for MacBook (Apple Silicon), fallback to CPU
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def get_modality(path):
    """
    Detects modality based on AMOS ID logic:
    CT: 0001-0500 | MRI: 0501-0600
    """
    filename = os.path.basename(path)
    # Extract numbers from filename (e.g., 'amos_0505.nii.gz' -> 505)
    try:
        case_id = int(''.join(filter(str.isdigit, filename)))
        if 501 <= case_id <= 600:
            return "mri"
    except ValueError:
        print("⚠️ Could not parse Case ID from filename. Defaulting to CT.")
    
    return "ct"

def visualize():
    # 1️⃣ Identify Modality
    modality = get_modality(IMAGE_PATH)
    print(f"📡 Detected Modality: {modality.upper()}")

    # 2️⃣ Build Dynamic Transforms
    if modality == "ct":
        # CT Windowing for Liver/Tumor
        intensity_transform = ScaleIntensityRanged(
            keys=["image"], a_min=-200, a_max=250, 
            b_min=0.0, b_max=1.0, clip=True
        )
    else:
        # MRI Z-Score Normalization
        intensity_transform = NormalizeIntensityd(keys=["image"], nonzero=True)

    transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(1.5, 1.5, 1.5), mode="bilinear"),
        intensity_transform, # <--- The modality-specific step
        EnsureTyped(keys=["image"]),
        ToTensord(keys=["image"]),
    ])

    # 3️⃣ Load Model
    print(f"📦 Loading weights to {DEVICE}...")
    model = get_lits_model().to(DEVICE)
    # map_location is essential when moving weights between Linux/CUDA and Mac/MPS
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # 4️⃣ Preprocess & Inference
    data = transforms({"image": IMAGE_PATH})
    inputs = data["image"].unsqueeze(0).to(DEVICE)

    print("🧠 Running Inference (Sliding Window)...")
    with torch.no_grad():
        # sw_batch_size=1 is safer for MacBook Air memory limits
        output = sliding_window_inference(inputs, (128, 128, 128), 1, model, overlap=0.25)
        seg = torch.argmax(output, dim=1).detach().cpu().numpy()[0]
        img = inputs.cpu().numpy()[0, 0]

    # 5️⃣ Visualization Logic
    # Find the slice with the most segmentation (Liver + Tumor)
    if np.max(seg) > 0:
        slice_sums = np.sum(seg, axis=(0, 1))
        slice_idx = np.argmax(slice_sums)
    else:
        slice_idx = img.shape[2] // 2

    plt.figure(figsize=(12, 6), facecolor='black')
    
    # Left: Original Scan
    plt.subplot(1, 2, 1)
    plt.imshow(img[:, :, slice_idx], cmap="gray")
    plt.title(f"{modality.upper()} Scan - Slice {slice_idx}", color="white")
    plt.axis("off")
    
    # Right: RGB Segmentation Overlay
    plt.subplot(1, 2, 2)
    rgb = np.zeros((*seg[:, :, slice_idx].shape, 3))
    rgb[seg[:, :, slice_idx] == 1] = [0, 1, 0] # Liver = Green
    rgb[seg[:, :, slice_idx] == 2] = [1, 0, 0] # Tumor = Red
    
    plt.imshow(rgb)
    plt.title("Prediction: Green=Liver, Red=Tumor", color="white")
    plt.axis("off")
    
    # Save result
    os.makedirs("outputs", exist_ok=True)
    save_path = f"outputs/test_{modality}_{os.path.basename(IMAGE_PATH)}.png"
    plt.savefig(save_path, bbox_inches='tight', facecolor='black')
    print(f"✅ Success! Visualization saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    if os.path.exists(IMAGE_PATH):
        visualize()
    else:
        print(f"❌ Error: Image not found at {IMAGE_PATH}")