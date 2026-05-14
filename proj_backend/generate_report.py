import torch
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, 
    Orientationd, EnsureTyped, ToTensord
)

# Import modular components
from amos_dataset import AmosNormalizeD
from kits_model import get_model as get_kits_model
from model_lits import get_lits_model
from model import get_model as get_amos_model # <--- Ensure this points to your 15-class SwinUNETR/SegResNet
from xai_engine import get_gradcam_heatmap
from radiomics_engine import extract_radiomics

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda")

def get_expert_config(organ, modality):
    if organ.lower() == "liver":
        return {
            "model_func": get_lits_model, 
            "weights": "outputs/best_lits_model.pth", 
            "amos_label": 6, # AMOS Liver ID
            "name": "Liver", "modality": modality
        }
    else: # Kidney
        return {
            "model_func": get_kits_model, 
            "weights": "outputs/best_kits_model.pth", 
            "amos_label": [2, 3], # AMOS Left/Right Kidney IDs
            "name": "Kidney", "modality": modality
        }

def generate_full_clinical_report(image_path, organ, modality, output_dir="./test_results"):
    cfg = get_expert_config(organ, modality)
    os.makedirs(output_dir, exist_ok=True)
    
    # 1️⃣ Preprocessing
    transforms = Compose([
        LoadImaged(keys=["image"]), EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(1.5, 1.5, 1.5), mode="bilinear"),
        AmosNormalizeD(keys=["image"]), EnsureTyped(keys=["image"]), ToTensord(keys=["image"]),
    ])

    data = transforms({"image": image_path, "modality": modality, "image_meta": image_path})
    inputs = data["image"].unsqueeze(0).to(DEVICE)

    # 2️⃣ STEP 1: Anatomical Guardrail (AMOS)
    print(f"🛡️ Using AMOS to define anatomical guardrail for {cfg['name']}...")
    amos_model = get_amos_model(DEVICE)
    amos_model.load_state_dict(torch.load("outputs/best_amos_model_new.pth", map_location=DEVICE))
    amos_model.eval()

    with torch.no_grad():
        amos_out = sliding_window_inference(inputs, (128, 128, 128), 1, amos_model)
        amos_mask = torch.argmax(amos_out, dim=1).cpu().numpy()[0]
    
    # Create a binary "allowed" mask for the organ
    if isinstance(cfg["amos_label"], list):
        allowed_region = np.isin(amos_mask, cfg["amos_label"])
    else:
        allowed_region = (amos_mask == cfg["amos_label"])

    # 3️⃣ STEP 2: Expert Pathology Segmentation
    print(f"🧠 Running {cfg['name']} Expert Model...")
    expert_model = cfg["model_func"]().to(DEVICE)
    expert_model.load_state_dict(torch.load(cfg["weights"], map_location=DEVICE))
    expert_model.eval()

    with torch.no_grad():
        output = sliding_window_inference(inputs, (128, 128, 128), 1, expert_model, overlap=0.25)
        expert_mask = torch.argmax(output, dim=1).cpu().numpy()[0]

    # 4️⃣ STEP 3: Hierarchical Intersection (THE FIX)
    # The final prediction is: Organ = AMOS results, Tumor = Expert results ONLY inside AMOS area
    final_seg = np.zeros_like(expert_mask)
    final_seg[allowed_region] = 1 # Mark healthy organ based on AMOS
    
    # Tumor (2) must be detected by Expert AND be inside the allowed anatomical region
    final_tumor_mask = (expert_mask == 2) & allowed_region
    final_seg[final_tumor_mask] = 2
    
    img_display = inputs.detach().cpu().numpy()[0, 0]

    # 5️⃣ Validation & Stats
    stats = extract_radiomics(img_display, final_seg, tumor_id=2)
    has_significant_tumor = True if stats else False
    
    print("🔥 Generating Grad-CAM...")
    heatmap = get_gradcam_heatmap(expert_model, inputs, final_seg, target_class=2)

    # 6️⃣ Slicing & Visualization
    if has_significant_tumor:
        slice_idx = np.argmax(np.sum(final_seg == 2, axis=(0, 1)))
    elif np.sum(final_seg == 1) > 0:
        slice_idx = np.argmax(np.sum(final_seg == 1, axis=(0, 1)))
    else:
        slice_idx = img_display.shape[2] // 2

    fig, ax = plt.subplots(1, 3, figsize=(18, 6), facecolor='black')
    plt.subplots_adjust(wspace=0.05)

    ax[0].imshow(img_display[:, :, slice_idx], cmap='gray')
    ax[0].set_title("Input CT Scan", color="white")
    ax[0].axis('off')

    # AI Prediction
    ax[1].imshow(img_display[:, :, slice_idx], cmap='gray')
    rgb_mask = np.zeros((*final_seg[:,:,slice_idx].shape, 3))
    organ_color = [0, 1, 0] if cfg['name'] == "Liver" else [0, 0, 1] 
    rgb_mask[final_seg[:,:,slice_idx] == 1] = organ_color
    if has_significant_tumor:
        rgb_mask[final_seg[:,:,slice_idx] == 2] = [1, 0, 0]
    ax[1].imshow(rgb_mask, alpha=0.6 if np.max(final_seg)>0 else 0)
    ax[1].set_title(f"AI Prediction (Red=Tumor)", color="white")
    ax[1].axis('off')

    # Grad-CAM (Only show if real tumor exists)
    ax[2].imshow(img_display[:, :, slice_idx], cmap='gray')
    if has_significant_tumor:
        masked_heatmap = np.ma.masked_where(heatmap[:, :, slice_idx] < 0.5, heatmap[:, :, slice_idx])
        ax[2].imshow(masked_heatmap, cmap='jet', alpha=0.7)
    ax[2].set_title("XAI: Tumor Evidence", color="white")
    ax[2].axis('off')

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"report_{os.path.basename(image_path)}.png")
    plt.savefig(save_path, facecolor='black', dpi=200)
    plt.close(fig)
    
    return {"report_image_path": save_path, "has_tumor": has_significant_tumor, "stats": stats}