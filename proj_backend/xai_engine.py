import torch
import torch.nn.functional as F
import numpy as np
from monai.visualize import GradCAM

def get_gradcam_heatmap(model, input_tensor, seg_mask, target_class=2):
    """
    High-Performance Focused Grad-CAM for 3D Medical Imaging.
    NOTE: Do NOT import this file inside itself.
    """
    
    # Force CPU for Grad-CAM calculations on Mac to prevent memory errors
    device = "cpu"
    model.to(device)
    input_tensor = input_tensor.to(device)
    
    # 1️⃣ Check if the tumor actually exists
    coords = np.argwhere(seg_mask == target_class)
    
    if coords.size == 0:
        print("ℹ️ Healthy scan: No tumor detected for XAI.")
        return np.zeros(input_tensor.shape[2:], dtype=float)

    # 2️⃣ Calculate bounding box to focus AI 'Attention'
    z_min, y_min, x_min = coords.min(axis=0)
    z_max, y_max, x_max = coords.max(axis=0)
    
    z_mid, y_mid, x_mid = (z_min + z_max)//2, (y_min + y_max)//2, (x_min + x_max)//2
    
    # Crop box around center
    z_start, z_end = max(0, z_mid-64), min(input_tensor.shape[2], z_mid+64)
    y_start, y_end = max(0, y_mid-64), min(input_tensor.shape[3], y_mid+64)
    x_start, x_end = max(0, x_mid-64), min(input_tensor.shape[4], x_mid+64)

    # 3️⃣ Extract ROI
    roi_input = input_tensor[:, :, z_start:z_end, y_start:y_end, x_start:x_end]
    
    # Ensure ROI dimensions are divisible by 16 for SegResNet
    d, h, w = roi_input.shape[2:]
    pad_d = (16 - d % 16) % 16
    pad_h = (16 - h % 16) % 16
    pad_w = (16 - w % 16) % 16
    roi_input = F.pad(roi_input, (0, pad_w, 0, pad_h, 0, pad_d))

    # 4️⃣ Run Grad-CAM
    cam = GradCAM(nn_module=model, target_layers="conv_final")
    
    print(f"🔥 Computing Grad-CAM... ROI: {roi_input.shape}")
    
    with torch.enable_grad():
        roi_heatmap = cam(x=roi_input, class_idx=target_class)
    
    # 5️⃣ Process Heatmap
    roi_data = roi_heatmap.detach().cpu().numpy()[0, 0]
    roi_data = roi_data[:d, :h, :w] # Remove padding
    
    # Local Normalization for high contrast
    if roi_data.max() > 0:
        roi_data = (roi_data - roi_data.min()) / (roi_data.max() - roi_data.min() + 1e-8)
    
    # 6️⃣ Re-embed into full volume
    full_heatmap = np.zeros(input_tensor.shape[2:], dtype=float)
    full_heatmap[z_start:z_end, y_start:y_end, x_start:x_end] = roi_data
    
    return full_heatmap