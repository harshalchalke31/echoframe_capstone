import torch
import torch.nn as nn
import torch.optim as optim
import csv
import os
from src.utils import BCEDiceLoss, dice_coefficient
from tqdm import tqdm
import torch.nn.functional as F

def train_UNet3D_weak_supervision(
    model,
    train_loader,
    valid_loader,
    device,
    num_epochs=500,
    lr=1e-3,
    log_path='./logs/train_log3d.csv',
    model_path='./models/best_model_3d.pth',
    patience=50,
    temporal_lambda=0.01  # Weight for temporal smoothness (needs tuning)
):
    """
    Trains a 3D UNet in a weakly supervised manner, with:
      - Loss computed only on the annotated center frame (partial supervision).
      - Optional temporal regularization to penalize abrupt changes across adjacent frames.
    Args:
        model: 3D UNet model
        train_loader: DataLoader for training
        valid_loader: DataLoader for validation
        device: "cpu" or "cuda"
        num_epochs: total epochs
        lr: learning rate
        log_path: CSV log file path
        model_path: saved model path
        patience: early stopping patience
        temporal_lambda: weight for the temporal regularization term
    """

    criterion = BCEDiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-7)

    best_loss = float('inf')
    patience_counter = 0
    model.to(device)

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    with open(log_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Train Loss", "Valid Loss", "Valid Dice"])

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        total_train_samples = 0

        for images, masks,_ in tqdm(train_loader, desc=f"Epoch {epoch+1} [Training]", leave=False):
            """
            images: shape [B, C, T, H, W]
            masks:  shape [B, 1, H, W]    (only center frame is labeled)
            """
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)      # shape [B, out_channels, T, H, W]

            # 1) Compute segmentation loss for the center frame only:
            center_idx = images.shape[2] // 2
            pred_center = outputs[:, :, center_idx]  # [B, out_channels, H, W]
            mask_center = masks                      # [B, 1,H, W]

            seg_loss = criterion(pred_center, mask_center)

            # 2) (Optional) Temporal regularization across frames
            #    using L1 difference between consecutive frames (logits).
            if temporal_lambda > 0:
                # shape [B, out_channels, T, H, W]
                b, c, t, h, w = outputs.shape
                temp_loss = 0.0
                # compare consecutive frames in [0..T-1]
                for frame_idx in range(1, t):
                    temp_loss += F.l1_loss(outputs[:, :, frame_idx],
                                           outputs[:, :, frame_idx - 1])
                # average across T-1 intervals
                temp_loss = temp_loss / (t - 1)
                total_loss = seg_loss + temporal_lambda * temp_loss
            else:
                total_loss = seg_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            train_loss += total_loss.item() * images.size(0)
            total_train_samples += images.size(0)

        train_loss /= total_train_samples

        # Validation
        model.eval()
        valid_loss = 0.0
        valid_dice = 0.0
        total_valid_samples = 0

        with torch.no_grad():
            for images, masks in tqdm(valid_loader, desc=f"Epoch {epoch+1} [Validation]", leave=False):
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                center_idx = images.shape[2] // 2

                pred_center = outputs[:, :, center_idx]
                mask_center = masks

                seg_loss = criterion(pred_center, mask_center)
                loss_val = seg_loss

                valid_loss += loss_val.item() * images.size(0)
                total_valid_samples += images.size(0)

                # Compute Dice at the center frame
                batch_dice = dice_coefficient(pred_center, mask_center)
                valid_dice += batch_dice * images.size(0)

        valid_loss /= total_valid_samples
        valid_dice /= total_valid_samples
        scheduler.step()

        with open(log_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, train_loss, valid_loss, valid_dice])

        # Save best model
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), model_path)
            patience_counter = 0
        else:
            patience_counter += 1

        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} | Valid Loss: {valid_loss:.4f} | "
            f"Valid Dice: {valid_dice:.4f} | Patience: {patience_counter}"
        )

        if patience_counter >= patience:
            print("Early Stopping Triggered!")
            break
