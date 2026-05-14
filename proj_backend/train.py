import os
import torch
from tqdm import tqdm
from monai.data import PersistentDataset, DataLoader
from monai.losses import DiceLoss
from model import get_model
from monai.metrics import DiceMetric
from monai.utils import set_determinism
from monai.transforms import AsDiscrete

from dataset import get_amos_datalist, get_transforms

def train():
    set_determinism(seed=42)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"--- Training started on device: {device} ---")

    data_dir = "/Users/malavikavr/Desktop/Main_project/AMOS22" 

    # 1. Load Train and Val sets separately from their folders
    train_files = get_amos_datalist(data_dir, mode="Tr")
    val_files = get_amos_datalist(data_dir, mode="Va")

    print(f"Train samples: {len(train_files)} | Val samples: {len(val_files)}")

    cache_dir = os.path.join(os.getcwd(), "persistent_cache")
    os.makedirs(cache_dir, exist_ok=True)

    # 2. Datasets
    train_ds = PersistentDataset(data=train_files, transform=get_transforms(), cache_dir=cache_dir)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)

    val_ds = PersistentDataset(data=val_files, transform=get_transforms(), cache_dir=cache_dir)
    val_loader = DataLoader(val_ds, batch_size=1)

    # 3. Model, Loss, Optimizer
    model = get_model(device)
    loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(model.parameters(), 1e-3)
    
    # Setup metrics
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    post_pred = AsDiscrete(argmax=True, to_onehot=8)
    post_label = AsDiscrete(to_onehot=8)

    # 4. Loop
    max_epochs = 50 # Increased slightly for better results
    val_interval = 2
    best_metric = -1
    
    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for batch_data in progress_bar:
            inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        print(f"--- Epoch {epoch+1} Avg Loss: {epoch_loss/len(train_loader):.4f} ---")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                for val_data in val_loader:
                    v_in, v_lab = val_data["image"].to(device), val_data["label"].to(device)
                    v_out = model(v_in)
                    
                    # Compute Dice
                    v_out = [post_pred(i) for i in v_out]
                    v_lab = [post_label(i) for i in v_lab]
                    dice_metric(y_pred=v_out, y=v_lab)

                metric = dice_metric.aggregate().item()
                dice_metric.reset()
                print(f"Validation Dice: {metric:.4f}")

                if metric > best_metric:
                    best_metric = metric
                    torch.save(model.state_dict(), "best_amos_model.pth")
                    print("Best model saved!")

if __name__ == "__main__":
    # Clear cache if you changed dataset logic significantly
    # os.system("rm -rf persistent_cache") 
    train()