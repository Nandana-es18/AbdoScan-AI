import torch
from monai.networks.nets import SegResNet

def get_model():
    model = SegResNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=8,
        init_filters=32,
        blocks_down=[1, 2, 2, 4],
        blocks_up=[1, 1, 1],
    )
    return model

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing SegResNet on {device}...")
    try:
        model = get_model().to(device)
        dummy_input = torch.randn(1, 1, 96, 96, 96).to(device)
        with torch.no_grad():
            output = model(dummy_input)
        print("✅ Success! Model is ready.")
        print(f"✅ Output shape: {output.shape}")
    except Exception as e:
        print(f"❌ Failed: {e}")
