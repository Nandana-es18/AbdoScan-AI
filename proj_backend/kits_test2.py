import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd,
    Orientationd, ScaleIntensityRanged, NormalizeIntensityd, 
    ScaleIntensityd, EnsureTyped, ToTensord
)
from monai.inferers import sliding_window_inference
from kits_model import get_model # Ensure this is your 3-class SegResNet

# ==========================================================
# DEVICE SETUP
# ==========================================================
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("🚀 Using Apple Silicon GPU (MPS)")
    torch.backends.mps.fallback = True
else:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using Device: {DEVICE}")

# Example Path
TEST_IMAGE_PATH = "/Users/merinphilip/Desktop/ProjectAMOS/data/imageTs/imaging.nii.gz"
MODEL_WEIGHTS = "outputs/best_kits_model.pth"

def get_modality(image_path):
    """
    Detects modality based on Case ID.
    CT: 0001-0500 | MRI: 0501-0600
    """
    filename = os.path.basename(image_path)
    try:
        case_id_str = "".join(filter(str.isdigit, filename))
        case_id = int(case_id_str)
        if 501 <= case_id <= 600:
            return "mri"
    except:
        pass
    return "ct" # Default to CT

def run_inference(image_path, manual_modality=None):
    # 1️⃣ Detect Modality
    modality = manual_modality if manual_modality else get_modality(image_path)
    print(f"📡 Detected Modality: {modality.upper()}")

    # 2️⃣ Load Model
    print(f"📦 Loading model weights...")
    model = get_model().to(DEVICE)
    # map_location is critical for moving weights from Linux/CUDA to Mac/MPS
    state_dict = torch.load(MODEL_WEIGHTS, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()

    # 3️⃣ Define Transforms based on Modality
    scaling_transform = None
    if modality == "ct":
        scaling_transform = ScaleIntensityRanged(
            keys=["image"], a_min=-200, a_max=300, b_min=0.0, b_max=1.0, clip=True
        )
    else:
        # MRI Logic: Standardize intensity based on the scan's mean/std
        scaling_transform = NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True)

    test_transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(1.5, 1.5, 1.5), mode="bilinear"),
        scaling_transform, # Dynamic scaling
        EnsureTyped(keys=["image"], device=DEVICE),
        ToTensord(keys=["image"]),
    ])

    # 4️⃣ Preprocess & Inference
    print(f"🚀 Processing: {os.path.basename(image_path)}")
    data_dict = test_transforms({"image": image_path})
    input_tensor = data_dict["image"].unsqueeze(0).to(DEVICE).float()

    print("🧠 Running AI Segmentation...")
    with torch.no_grad():
        output = sliding_window_inference(
            input_tensor,
            roi_size=(128, 128, 128),
            sw_batch_size=1, # 1 is safer for MacBook Air RAM
            predictor=model
        )
        seg = torch.argmax(output, dim=1).cpu().numpy()[0]
        img = input_tensor.cpu().numpy()[0, 0]

    # 5️⃣ Visualization
    # Find best slice (where Kidney or Tumor is largest)
    slice_idx = img.shape[2] // 2
    if np.max(seg) > 0:
        slice_sums = np.sum(seg, axis=(0, 1))
        slice_idx = np.argmax(slice_sums)

    plt.figure(figsize=(12, 6), facecolor='black')
    plt.subplot(1, 2, 1)
    plt.imshow(img[:, :, slice_idx], cmap="gray")
    plt.title(f"Input {modality.upper()} Scan (Slice {slice_idx})", color="white")
    plt.axis("off")

    rgb = np.zeros((*seg[:, :, slice_idx].shape, 3))
    rgb[seg[:, :, slice_idx] == 1] = [0, 0, 1] # Kidney = Blue
    rgb[seg[:, :, slice_idx] == 2] = [1, 0, 0] # Tumor = Red

    plt.subplot(1, 2, 2)
    plt.imshow(rgb)
    plt.title("AI Prediction (Blue: Kidney, Red: Tumor)", color="white")
    plt.axis("off")

    os.makedirs("outputs", exist_ok=True)
    save_filename = f"inference_{modality}_{os.path.basename(image_path)}.png"
    plt.savefig(os.path.join("outputs", save_filename), bbox_inches='tight', facecolor='black')
    print(f"✅ Success! Saved to outputs/{save_filename}")
    plt.show()

if __name__ == "__main__":
    # Test a CT scan
    # run_inference(TEST_IMAGE_PATH, manual_modality="ct")
    
    # Test an MRI scan
    # run_inference("path/to/mri_scan.nii.gz", manual_modality="mri")
    
    if os.path.exists(TEST_IMAGE_PATH):
        run_inference(TEST_IMAGE_PATH)