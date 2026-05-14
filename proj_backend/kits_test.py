import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd,
    Orientationd, ScaleIntensityRanged, EnsureTyped, ToTensord
)
from monai.inferers import sliding_window_inference
from kits_model import get_model

# ==========================================================
# DEVICE SETUP FOR APPLE SILICON
# ==========================================================
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("🚀 Using Apple Silicon GPU (MPS)")
    torch.backends.mps.fallback = True
else:
    DEVICE = torch.device("cpu")
    print("⚠️ MPS not available, using CPU")

TEST_IMAGE_PATH = "/Users/merinphilip/Desktop/ProjectAMOS/data/imageTs/imaging1.nii.gz"
MODEL_WEIGHTS = "outputs/best_kits_model.pth"


def run_inference_on_file(image_path):

    # 1️⃣ Load Model
    print(f"📦 Loading model weights from {MODEL_WEIGHTS}...")
    model = get_model().to(DEVICE)

    state_dict = torch.load(MODEL_WEIGHTS, map_location=DEVICE)
    model.load_state_dict(state_dict)

    model.eval()

    # 2️⃣ Transforms (same as training)
    test_transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(1.5, 1.5, 1.5), mode="bilinear"),
        ScaleIntensityRanged(
            keys=["image"], a_min=-200, a_max=300,
            b_min=0.0, b_max=1.0, clip=True
        ),
        EnsureTyped(keys=["image"], device=DEVICE),
        ToTensord(keys=["image"]),
    ])

    # 3️⃣ Preprocess
    print(f"🚀 Preprocessing image: {os.path.basename(image_path)}")
    data_dict = test_transforms({"image": image_path})

    input_tensor = data_dict["image"].unsqueeze(0).to(DEVICE).float()

    # 4️⃣ Inference
    print("🧠 Running AI Segmentation...")
    with torch.no_grad():
        output = sliding_window_inference(
            input_tensor,
            roi_size=(128, 128, 128),
            sw_batch_size=2,   # 🔥 smaller for MPS memory
            predictor=model
        )

        seg = torch.argmax(output, dim=1).cpu().numpy()[0]
        img = input_tensor.cpu().numpy()[0, 0]

    # 5️⃣ Best slice selection
    if np.max(seg) == 2:
        slice_idx = np.argmax(np.sum(seg == 2, axis=(0, 1)))
        print(f"📍 Found tumor in slice index: {slice_idx}")

    elif np.max(seg) == 1:
        slice_idx = np.argmax(np.sum(seg == 1, axis=(0, 1)))
        print(f"📍 No tumor found, showing kidney slice: {slice_idx}")

    else:
        slice_idx = img.shape[2] // 2
        print(f"⚠️ No segmentation found, showing middle slice: {slice_idx}")

    # 6️⃣ Visualization
    plt.figure(figsize=(12, 6), facecolor='black')

    plt.subplot(1, 2, 1)
    plt.imshow(img[:, :, slice_idx], cmap="gray")
    plt.title("Input CT Scan", color="white")
    plt.axis("off")

    rgb = np.zeros((*seg[:, :, slice_idx].shape, 3))
    rgb[seg[:, :, slice_idx] == 1] = [0, 0, 1]
    rgb[seg[:, :, slice_idx] == 2] = [1, 0, 0]

    plt.subplot(1, 2, 2)
    plt.imshow(rgb)
    plt.title("AI Prediction (Blue: Kidney, Red: Tumor)", color="white")
    plt.axis("off")

    os.makedirs("outputs", exist_ok=True)
    save_filename = f"test_result_{os.path.basename(image_path)}.png"
    plt.savefig(os.path.join("outputs", save_filename),
                bbox_inches='tight', facecolor='black')

    print(f"✅ Success! Result saved to outputs/{save_filename}")
    plt.show()


if __name__ == "__main__":
    if os.path.exists(TEST_IMAGE_PATH):
        run_inference_on_file(TEST_IMAGE_PATH)
    else:
        print(f"❌ Error: File not found at {TEST_IMAGE_PATH}")