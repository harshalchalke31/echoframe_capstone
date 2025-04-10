import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from .utils import generate_random_mask, MaskedMSELoss, psnr, temporal_consistency_loss,ssim_score



def train_autoencoder_3d(
    model,
    train_loader,
    valid_loader,
    device: str,
    num_epochs: int = 100,
    lr: float = 1e-4,
    log_path: str = './logs/train_log_autoencoder.csv',
    model_path: str = './models/best_autoencoder.pth',
    patience: int = 30,
    use_masked_loss: bool = False,
    mask_ratio: float = 0.75,
    tdc_weight: float = 0.1
):
    # Loss selection
    criterion = MaskedMSELoss() if use_masked_loss else nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-7)

    best_loss = float('inf')
    patience_counter = 0
    model.to(device)

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    with open(log_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Train Loss", "Valid Loss", "Valid PSNR", "Valid SSIM", "Temporal Consistency"])

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        total_train_samples = 0

        for images in tqdm(train_loader, desc=f"Epoch {epoch+1} [Training]", leave=False):
            images = images.to(device)

            if use_masked_loss:
                mask = generate_random_mask(images.shape, mask_ratio=mask_ratio, device=device)
                inputs = images * mask
                outputs = model(inputs)
                recon_loss = criterion(outputs, images, mask)
            else:
                outputs = model(images)
                recon_loss = criterion(outputs, images)

            tdc_loss = temporal_consistency_loss(outputs, images)
            loss = recon_loss + tdc_weight * tdc_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = images.size(0)
            train_loss += loss.item() * batch_size
            total_train_samples += batch_size

        train_loss /= total_train_samples

        # --- Validation ---
        model.eval()
        valid_loss = 0.0
        total_valid_samples = 0
        total_psnr, total_ssim, total_temporal = 0.0, 0.0, 0.0

        with torch.no_grad():
            for images in tqdm(valid_loader, desc=f"Epoch {epoch+1} [Validation]", leave=False):
                images = images.to(device)

                if use_masked_loss:
                    mask = generate_random_mask(images.shape, mask_ratio=mask_ratio, device=device)
                    inputs = images * mask
                    outputs = model(inputs)
                    loss = criterion(outputs, images, mask)
                else:
                    outputs = model(images)
                    loss = criterion(outputs, images)

                batch_size = images.size(0)
                valid_loss += loss.item() * batch_size
                total_valid_samples += batch_size

                total_psnr += psnr(outputs, images).item() * batch_size
                total_ssim += ssim_score(outputs, images).item() * batch_size
                total_temporal += temporal_consistency_loss(outputs, images).item() * batch_size

        valid_loss /= total_valid_samples
        avg_psnr = total_psnr / total_valid_samples
        avg_ssim = total_ssim / total_valid_samples
        avg_temporal = total_temporal / total_valid_samples
        scheduler.step()

        with open(log_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, train_loss, valid_loss, avg_psnr, avg_ssim, avg_temporal])

        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {train_loss:.6f} | "
              f"Valid Loss: {valid_loss:.6f} | "
              f"PSNR: {avg_psnr:.2f} dB | SSIM: {avg_ssim:.4f} | "
              f"TDC: {avg_temporal:.6f} | Patience: {patience_counter}")

        if valid_loss < best_loss:
            torch.save(model.encoder.state_dict(), model_path)
            best_loss = valid_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early Stopping Triggered!")
            break
