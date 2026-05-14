import torch
import os
import matplotlib.pyplot as plt
from torch.amp import GradScaler, autocast
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete
from monai.data import decollate_batch
from tqdm import tqdm

from dataset import get_kits_dataloader
from model import get_model

# RTX 4000 Ada Optimization
torch.set_float32_matmul_precision('high')

def train():
    os.makedirs("outputs", exist_ok=True)
    device = torch.device("cuda")
    model = get_model().to(device)
    
    train_loader = get_kits_dataloader("./data", batch_size=1, phase="train")
    val_loader = get_kits_dataloader("./data", batch_size=1, phase="val")
    
    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    dice_metric = DiceMetric(include_background=False, reduction="mean_batch")
    post_label = AsDiscrete(to_onehot=3); post_pred = AsDiscrete(argmax=True, to_onehot=3)
    scaler = GradScaler("cuda")

    train_loss_history = []
    val_dice_k = []; val_dice_t = []; val_epochs = []
    best_dice = -1

    for epoch in range(200):
        model.train()
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            inputs, labels = batch["image"].to(device), batch["label"].to(device)
            optimizer.zero_grad()
            with autocast("cuda"):
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        train_loss_history.append(epoch_loss/len(train_loader))

        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs, val_labels = val_data["image"].to(device), val_data["label"].to(device)
                    with autocast("cuda"):
                        val_outputs = sliding_window_inference(val_inputs, (128, 128, 128), 4, model)
                    val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                    val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                    dice_metric(y_pred=val_outputs, y=val_labels)
                
                metric = dice_metric.aggregate()
                val_epochs.append(epoch)
                val_dice_k.append(metric[0].item()); val_dice_t.append(metric[1].item())
                mean_dice = torch.mean(metric).item()
                print(f" Kidney: {metric[0].item():.4f}, Tumor: {metric[1].item():.4f}")
                dice_metric.reset()

                if mean_dice > best_dice:
                    best_dice = mean_dice
                    torch.save(model.state_dict(), "outputs/best_kits_model.pth")

    # SAVE RESEARCH GRAPHS
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1); plt.plot(train_loss_history, label="Loss"); plt.title("Training Loss"); plt.legend()
    plt.subplot(1, 2, 2); plt.plot(val_epochs, val_dice_k, label="Kidney"); plt.plot(val_epochs, val_dice_t, label="Tumor")
    plt.title("Dice Accuracy"); plt.legend()
    plt.savefig("outputs/accuracy_graph.png")
    print("📈 Graphs saved to outputs/accuracy_graph.png")

if __name__ == "__main__":
    train()