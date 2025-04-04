from src.utils import BCEDiceLoss, dice_coefficient, DiceLoss
import torch
import torch.optim as optim
import csv
import torch.nn as nn
import os


def train_UNet(model, train_loader, valid_loader, device, num_epochs=500, lr=1e-3,
               log_path='./logs/train_log2.csv', model_path='./models/model1/best_model2.pth',patience=50):
    # using the combined BCE + Dice loss
    criterion = BCEDiceLoss()
    # adam with weight decay for regularization
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    # cosine annealing scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-7)

    best_loss = float('inf')
    patience_counter = 0
    model.to(device)
    # writing the log file
    with open(log_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Train Loss", "Valid Loss", "Valid Dice"])

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        total_train_samples = 0

        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            batch_size_curr = images.size(0)
            train_loss += loss.item() * batch_size_curr
            total_train_samples += batch_size_curr
        
        train_loss = train_loss / total_train_samples

        model.eval()
        valid_loss = 0.0
        total_valid_samples = 0
        valid_dice = 0.0

        with torch.no_grad():
            for images, masks in valid_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                batch_size_curr = images.size(0)
                valid_loss += loss.item() * batch_size_curr
                total_valid_samples += batch_size_curr

                # compute dice coefficient for the batch and accumulate
                batch_dice = dice_coefficient(outputs, masks)
                valid_dice += batch_dice * batch_size_curr

        valid_loss = valid_loss / total_valid_samples
        valid_dice = valid_dice / total_valid_samples
        
        # step the cosine annealing scheduler
        scheduler.step()

        with open(log_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, train_loss, valid_loss, valid_dice])

        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), model_path)
            patience_counter=0
        else:
            patience_counter+=1
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, "
              f"Valid Loss: {valid_loss:.4f}, Valid Dice: {valid_dice:.4f}, Patience: {patience_counter}")

        if patience_counter>=patience:
            print('Early Stopping Triggered!')
            break

