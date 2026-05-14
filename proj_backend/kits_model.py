import torch
from monai.networks.nets import SegResNet

def get_model():
    """
    SegResNet: Optimized for Kidney & Tumor Segmentation.
    3 channels: 0: Background, 1: Kidney, 2: Tumor.
    """
    return SegResNet(
        spatial_dims=3,
        init_filters=32,
        in_channels=1,
        out_channels=3, 
        dropout_prob=0.2,
        blocks_down=[1, 2, 2, 4],
        blocks_up=[1, 1, 1]
    )

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Testing SegResNet on {device}...")
    try:
        model = get_model().to(device)
        dummy = torch.randn(1, 1, 128, 128, 128).to(device)
        with torch.no_grad():
            out = model(dummy)
        print(f"✅ Success! Output shape: {out.shape}")
    except Exception as e:
        print(f"❌ Failed: {e}")
