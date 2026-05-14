import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, 
    Spacingd, Resized, ToTensord
)
from monai.inferers import sliding_window_inference
from model import get_model
from dataset import AmosNormalizeD

def visualize_test_result():
    # 1. Setup Device and Paths
    device = torch.device("cpu")
    model_path = "best_amos_model.pth"
    test_dir = "/Users/merinphilip/Desktop/proj_backend/AMOS22/imageTs" # Ensure this path is correct
    output_dir = "./test_results"
    os.makedirs(output_dir, exist_ok=True)

    # 2. Define the 8 Organ Labels (for the Legend)
    organ_names = [
        "Background", "Spleen", "R. Kidney", "L. Kidney", 
        "Gallbladder", "Pancreas", "Liver", "Stomach"
    ]
    num_classes = len(organ_names)

    # 3. Load the Model
    # Important: get_model must now return a UNet with out_channels=8
    model = get_model(device)
    if not os.path.exists(model_path):
        print(f"Error: Could not find {model_path}. Did you finish training?")
        return
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Model loaded successfully.")

    # 4. Get Test Files
    test_images = sorted(glob.glob(os.path.join(test_dir, "*.nii.gz")))
    if not test_images:
        print(f"No images found in {test_dir}!")
        return

    # Pick the first image in the folder for testing
    test_case = test_images[0] 
    print(f"Processing case: {os.path.basename(test_case)}")

    # 5. Preprocessing Transforms (Inference Mode)
    test_transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(1.5, 1.5, 2.0), mode="bilinear"),
        AmosNormalizeD(keys=["image"]),
        Resized(keys=["image"], spatial_size=[128, 128, 64]),
        ToTensord(keys=["image"]),
    ])

    # Prepare data dictionary (including image_meta for CT/MRI check)
    data_dict = {"image": test_case, "image_meta": test_case}
    data_processed = test_transforms(data_dict)
    
    input_tensor = data_processed["image"].unsqueeze(0).to(device)

    # 6. Run Inference
    with torch.no_grad():
        # Using sliding window inference for better boundary results
        output = sliding_window_inference(input_tensor, (128, 128, 64), 4, model)
        # Convert output probabilities to class IDs (0 to 7)
        prediction = torch.argmax(output, dim=1).detach().cpu().numpy()[0]

    # 7. Visualization Setup
    image_np = data_processed["image"][0].cpu().numpy()
    
    # Selection of 3 slices (Axial view)
    slices = [20, 32, 45]
    
    fig, axes = plt.subplots(len(slices), 2, figsize=(12, 18))
    
    # Set up the Colormap and Legend
    cmap = plt.get_cmap("nipy_spectral")
    # Generate colors based on our 8 classes
    colors = [cmap(i / (num_classes - 1)) for i in range(num_classes)]
    legend_handles = [mpatches.Patch(color=colors[i], label=f"{i}: {organ_names[i]}") for i in range(1, num_classes)]

    for i, s in enumerate(slices):
        # Plot Original Scan (Grayscale)
        axes[i, 0].imshow(image_np[:, :, s], cmap="gray")
        axes[i, 0].set_title(f"Original Scan - Slice {s}")
        axes[i, 0].axis("off")
        
        # Plot Prediction Mask (Spectral Colors)
        # vmin/vmax ensures color 7 is always the same color regardless of what's in the slice
        axes[i, 1].imshow(prediction[:, :, s], cmap="nipy_spectral", vmin=0, vmax=num_classes-1)
        axes[i, 1].set_title(f"Predicted Segmentation - Slice {s}")
        axes[i, 1].axis("off")

    # Place legend on the right side
    fig.legend(handles=legend_handles, loc='center right', bbox_to_anchor=(1.1, 0.5), fontsize='medium')
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"prediction_{os.path.basename(test_case)}.png")
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Visualization saved to: {save_path}")
    plt.show()

if __name__ == "__main__":
    visualize_test_result()