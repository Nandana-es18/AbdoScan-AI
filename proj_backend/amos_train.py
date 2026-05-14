import torch
import os
import numpy as np
from torch.cuda.amp import GradScaler, autocast
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete
from amos_dataset import get_dataloader
from amos_model import get_model
from tqdm import tqdm

DATA_DIR = "./data"

# ✅ Proper Mac Device Handling
DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

VAL_INTERVAL = 2
NUM_CLASSES = 8

ORGAN_NAMES = [
    "Spleen", "R.Kidney", "L.Kidney",
    "Gallbladder", "Pancreas", "Liver", "Stomach"
]

def train():
    os.makedirs("outputs", exist_ok=True)
    model = get_model().to(DEVICE)

    train_loader = get_dataloader(DATA_DIR, batch_size=1, phase="train")
    val_loader = get_dataloader(DATA_DIR, batch_size=1, phase="val")

    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    # ✅ AMP only if CUDA
    use_amp = DEVICE.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    dice_metric = DiceMetric(include_background=False, reduction="mean_batch")
    hd95_metric = HausdorffDistanceMetric(include_background=False, reduction="mean_batch", percentile=95)

    post_label = AsDiscrete(to_onehot=NUM_CLASSES)
    post_pred = AsDiscrete(argmax=True, to_onehot=NUM_CLASSES)

    best_dice = -1

    print(f"Using device: {DEVICE}")
    print("Starting Training...")

    for epoch in range(200):
        model.train()
        epoch_loss = 0

        for batch_data in tqdm(train_loader, desc=f"Epoch {epoch}"):
            inputs = batch_data["image"].to(DEVICE)
            labels = batch_data["label"].to(DEVICE)

            optimizer.zero_grad()

            with autocast(enabled=use_amp):
                outputs = model(inputs)
                loss = loss_function(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        # Validation
        if (epoch + 1) % VAL_INTERVAL == 0:
            model.eval()
            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs = val_data["image"].to(DEVICE)
                    val_labels = val_data["label"].to(DEVICE)

                    with autocast(enabled=use_amp):
                        val_outputs = sliding_window_inference(
                            val_inputs, (96, 96, 96), 4, model
                        )

                    val_outputs = [post_pred(i) for i in val_outputs]
                    val_labels = [post_label(i) for i in val_labels]

                    dice_metric(y_pred=val_outputs, y=val_labels)

                metric_batch = dice_metric.aggregate()
                mean_dice = torch.mean(metric_batch).item()
                dice_metric.reset()

                if mean_dice > best_dice:
                    best_dice = mean_dice
                    torch.save(model.state_dict(), "outputs/best_amos_model_new.pth")
                    print(f"New Best Dice: {mean_dice:.4f} - Saved!")

    print("\nTraining Complete.")
