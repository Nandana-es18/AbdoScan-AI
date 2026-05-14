import torch
from monai.networks.nets import SegResNet

def get_lits_model():
    """
    SegResNet Optimized for Liver/Tumor Segmentation.
    3 output channels: 0:BG, 1:Liver, 2:Tumor.
    """
    model = SegResNet(
        spatial_dims=3,
        init_filters=32,
        in_channels=1,
        out_channels=3, 
        dropout_prob=0.2,
        blocks_down=[1, 2, 2, 4],
        blocks_up=[1, 1, 1],
    )
    return model

# --- DIAGNOSTIC BLOCK ---
if __name__ == "__main__":
    print("🖥️  Testing LiTS Model Initialization...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📍 Using Device: {device}")
    
    try:
        # 1. Initialize Model
        model = get_lits_model().to(device)
        print("✅ Model created and moved to GPU successfully!")
        
        # 2. Test Forward Pass (Patch size 128x128x128)
        # This checks if the 20GB VRAM can handle the High-Res patch
        dummy_input = torch.randn(1, 1, 128, 128, 128).to(device)
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"✅ Forward pass successful!")
        print(f"📦 Output shape: {output.shape} (Expected: [1, 3, 128, 128, 128])")
        print("🚀 Workstation is ready for Liver/Tumor training.")
        
    except Exception as e:
        print(f"❌ Error: {e}")