import torch.nn as nn
import torch
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path 

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss,self).__init__()
    
    def forward(self,pred,target):
        smooth = 1e-6
        pred = torch.sigmoid(pred)
        intersection = (pred*target).sum()
        return 1- ((2.* intersection +smooth)/(pred.sum()+target.sum()+smooth))

class BCEDiceLoss(nn.Module):
    def __init__(self, weight=None, smooth=1e-5):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss(weight=weight)  # works with raw logits
        self.smooth = smooth

    def forward(self, inputs, targets):
        # compute BCE loss
        bce_loss = self.bce(inputs, targets)
        
        # compute Dice loss, apply sigmoid to get probabilities first
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        
        return bce_loss + dice_loss

def dice_coefficient(preds, targets, smooth=1e-5, threshold=0.5):
    # apply sigmoid to logits and then threshold to get binary masks
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()
    
    # flatten the tensors
    preds = preds.view(-1)
    targets = targets.view(-1)
    
    intersection = (preds * targets).sum()
    dice = (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)
    return dice.item()

def plot_loss_curves(log_path:Path):
    df = pd.read_csv(log_path)  

    # extract columns
    epochs = df['Epoch']
    train_loss = df['Train Loss']
    valid_loss = df['Valid Loss']
    valid_dice = df['Valid Dice']


    plt.figure(figsize=(10, 4))

    # --- Subplot 1: Loss curves ---
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, valid_loss, label='Valid Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss')
    plt.legend()
    plt.grid(True)

    # --- Subplot 2: Validation Dice ---
    plt.subplot(1, 2, 2)
    plt.plot(epochs, valid_dice, label='Valid Dice', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Dice')
    plt.title('Validation Dice')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()