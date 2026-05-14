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

from dataset_lits import get_lits_dataloader
from model_lits import get_lits_model

# Optimizations for RTX 4000 Ada
torch.set_float32_matmul_precision('high')

def train():
    os.makedirs("outputs_lits", exist_ok=True)
    device = torch.device("cuda")
    model = get_lits_model().to(device)
    
    train_loader = get_lits_dataloader("./data", batch_size=1, phase="train")
    val_loader = get_lits_dataloader("./data", batch_size=1, phase="val")
    
    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    dice_metric = DiceMetric(include_background=False, reduction="mean_batch")
    post_label = AsDiscrete(to_onehot=3); post_pred = AsDiscrete(argmax=True, to_onehot=3)
    scaler = GradScaler("cuda")

    history = {"loss": [], "liver_dice": [], "tumor_dice": [], "epoch": []}
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
        
        history["loss"].append(epoch_loss/len(train_loader))

        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                for val_data in val_loader:
                    v_in, v_lbl = val_data["image"].to(device), val_data["label"].to(device)
                    with autocast("cuda"):
                        v_out = sliding_window_inference(v_in, (128, 128, 128), 4, model)
                    v_out = [post_pred(i) for i in decollate_batch(v_out)]
                    v_lbl = [post_label(i) for i in decollate_batch(v_lbl)]
                    dice_metric(y_pred=v_out, y=v_lbl)
                
                metric = dice_metric.aggregate()
                history["epoch"].append(epoch)
                history["liver_dice"].append(metric[0].item())
                history["tumor_dice"].append(metric[1].item())
                mean_dice = torch.mean(metric).item()
                print(f" - Liver: {metric[0].item():.4f} | Tumor: {metric[1].item():.4f}")
                dice_metric.reset()

                if mean_dice > best_dice:
                    best_dice = mean_dice
                    torch.save(model.state_dict(), "outputs_lits/best_lits_model.pth")
                    print("⭐ New Best Saved")

    # Final Graphing for Paper
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1); plt.plot(history["loss"], color='red'); plt.title("Training Loss")
    plt.subplot(1, 2, 2); 
    plt.plot(history["epoch"], history["liver_dice"], label="Liver")
    plt.plot(history["epoch"], history["tumor_dice"], label="Tumor")
    plt.title("Dice Score Accuracy"); plt.legend()
    plt.savefig("outputs_lits/research_results.png")
    print("📈 Graphs saved to outputs_lits/research_results.png")

if __name__ == "__main__":
    train()