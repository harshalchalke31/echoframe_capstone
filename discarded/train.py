import torch
import torch.nn as nn
import torch.optim as optim
import csv
import os
from src.utils import BCEDiceLoss, dice_coefficient
from tqdm import tqdm


def train_UNet3D_weak_supervision(model, train_loader, valid_loader, device, num_epochs=500,
                                  lr=1e-3, log_path='./logs/train_log3d.csv',
                                  model_path='./models/best_model_3d.pth', patience=50):

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

        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1} [Training]", leave=False):
            images, masks = images.to(device), masks.to(device)  # masks: [B, 1, H, W]

            outputs = model(images)  # [B, 1, T, H, W]
            center = images.shape[2] // 2
            pred_center = outputs[:, :, center]       # [B, 1, H, W]
            mask_center = masks                       # [B, 1, H, W]

            loss = criterion(pred_center, mask_center)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
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
                center = images.shape[2] // 2
                pred_center = outputs[:, :, center]
                mask_center = masks

                loss = criterion(pred_center, mask_center)
                valid_loss += loss.item() * images.size(0)
                total_valid_samples += images.size(0)

                batch_dice = dice_coefficient(pred_center, mask_center)
                valid_dice += batch_dice * images.size(0)

        valid_loss /= total_valid_samples
        valid_dice /= total_valid_samples
        scheduler.step()

        with open(log_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, train_loss, valid_loss, valid_dice])

        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), model_path)
            patience_counter = 0
        else:
            patience_counter += 1

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | "
              f"Valid Loss: {valid_loss:.4f} | Valid Dice: {valid_dice:.4f} | "
              f"Patience: {patience_counter}")

        if patience_counter >= patience:
            print("Early Stopping Triggered!")
            break
