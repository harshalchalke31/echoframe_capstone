import torch.nn as nn
import torch
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import cv2
import pandas as pd
import random
import torchvision.transforms.functional as TF

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

def plot_loss_curves(log_path:Path,suptitle:str='Model Performance'):
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
    
    plt.suptitle(suptitle, fontsize=16)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


#------------------Helper code for testing-------------------#

PX_TO_CM = 0.14

def closest_point(point, array):
    diff = array - point
    dist_sq = np.einsum('ij,ij->i', diff, diff)
    return np.argmin(dist_sq), dist_sq

def farthest_point(point, array):
    diff = array - point
    dist_sq = np.einsum('ij,ij->i', diff, diff)
    return np.argmax(dist_sq), dist_sq

def label_points(p1, p2, p3):
    """
    Among the three corners (p1, p2, p3), the apex is the corner
    that is farthest from the other two corners.
    """
    d12 = np.linalg.norm(p1 - p2)
    d23 = np.linalg.norm(p2 - p3)
    d13 = np.linalg.norm(p1 - p3)

    # The corner forming the smallest side is the "mitral valve side."
    # The remaining corner is apex.
    if d12 < d23 and d12 < d13:
        apex = p3
        mv_1 = p1
        mv_2 = p2
    elif d23 < d12 and d23 < d13:
        apex = p1
        mv_1 = p3
        mv_2 = p2
    else:
        apex = p2
        mv_1 = p1
        mv_2 = p3
    return apex, mv_1, mv_2

def simpsons_biplane(cnt, frame):
    """
    Returns volume in mL using Simpson’s single-plane formula:
       V (cm^3) = 8 * (Area_cm^2)^2 / (3 * pi * Length_cm)
    where:
      - Area_cm^2 is area in cm^2
      - Length_cm is the apex-to-base length in cm
    """
    # Minimal enclosing triangle
    area_dummy, tri = cv2.minEnclosingTriangle(cnt)
    tri = tri.reshape(-1,2)  # shape (3,2)

    # For each triangle corner, get the closest contour point
    idx, _ = closest_point(tri[0], cnt[:, 0, :])
    bp1 = cnt[idx, 0, :]
    idx, _ = closest_point(tri[1], cnt[:, 0, :])
    bp2 = cnt[idx, 0, :]
    idx, _ = closest_point(tri[2], cnt[:, 0, :])
    bp3 = cnt[idx, 0, :]

    # Identify apex and MV corners
    apex, mv1, mv2 = label_points(bp1, bp2, bp3)
    mid_mv = (mv1 + mv2) / 2.0

    # Build a line from apex -> mid_mv
    apex_slope = apex - mid_mv
    length_slope = np.linalg.norm(apex_slope)
    if length_slope < 1e-6:
        return 0.0
    apex_slope /= length_slope

    # draw a line that exceeds the image diagonal
    h, w = frame.shape[:2]
    diag = int(np.sqrt(h**2 + w**2))
    line_len = diag * 2

    start_pt = (apex - line_len * apex_slope).astype(int)
    end_pt   = (mid_mv + line_len * apex_slope).astype(int)

    blank = np.zeros_like(frame, dtype=np.uint8)
    contour_mask = cv2.drawContours(blank.copy(), [cnt], 0, 1, 1)

    # keep thickening the line until we find at least two intersections
    # far enough apart (>= ~0.9 * MV diameter).
    annulus_diam = np.linalg.norm(mv1 - mv2)
    width = 2
    dist = 0

    while True:
        test_line = cv2.line(blank.copy(),
                             tuple(start_pt),
                             tuple(end_pt),
                             1, width)
        intersects = np.logical_and(contour_mask, test_line)
        pts = np.column_stack(np.where(intersects == 1))  # shape (N,2)

        if pts.size == 0:
            # No intersections
            width += 1
            if width > diag * 2:
                return 0.0
            continue

        # Find the one furthest from apex
        # pts is in (row, col) => flip for (x,y)
        idx_far, arr_far = farthest_point(np.flip(apex), pts)
        dist = arr_far[idx_far]
        if (len(pts) >= 2) and (dist >= 0.9 * annulus_diam):
            # Good enough
            break
        width += 1
        if width > diag * 2:
            return 0.0

    # The farthest intersection from apex is the "base intercept"
    intercept = pts[idx_far]  # (row,col)
    intercept = np.flip(intercept)  # => (x,y)

    # measure apex->intercept in px, convert to cm
    lv_length_px = np.linalg.norm(apex - intercept)
    lv_length_cm = lv_length_px * PX_TO_CM

    # contour area in px^2 => cm^2
    mask_area_px = cv2.contourArea(cnt)
    mask_area_cm2 = mask_area_px * (PX_TO_CM**2)

    # Simpson’s single-plane formula => volume in cm^3
    # which is equal to mL
    if lv_length_cm < 1e-6:
        return 0.0
    volume_mL = (8.0 * mask_area_cm2**2) / (3.0 * np.pi * lv_length_cm)
    return volume_mL

def compute_volume_from_mask(mask_np):
    """
    mask_np: shape (H,W) in {0,1}.
    Returns volume in mL via single-plane Simpson’s method.
    """
    mask_255 = (mask_np * 255).astype(np.uint8)
    # Find largest external contour
    cnts, _ = cv2.findContours(mask_255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return 0.0
    max_cnt = max(cnts, key=cv2.contourArea)

    dummy = np.zeros_like(mask_255)
    vol_ml = simpsons_biplane(max_cnt, dummy)
    return vol_ml

def test_performance(model,test_loader,test_df,device,model_path):
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()

    criterion = BCEDiceLoss() 

    test_loss = 0.0
    test_dice = 0.0
    total_pixels = 0
    total_samples = 0

    # store EF predictions and ground truths for regression metrics:
    efs_pred = []
    efs_true = []

    # Loop over test_loader
    with torch.no_grad():
        for images, masks, file_names in test_loader:
            images = images.to(device)
            masks  = masks.to(device)
            batch_size = images.shape[0]

            # Forward pass
            outputs = model(images)  # shape [B,1,H,W]
            
            loss = criterion(outputs, masks)
            test_loss += loss.item() * batch_size

            dice_val = dice_coefficient(outputs, masks)
            test_dice += dice_val * batch_size


            pred_bin = (outputs.cpu().numpy() > 0.5).astype(np.uint8)  # shape [B,1,H,W]

            # test_loader yields pairs of frames for the same video (ED & ES),
            volumes = []
            for i in range(batch_size):
                mask_pred_i = pred_bin[i,0]  # shape (H,W)
                vol_i = compute_volume_from_mask(mask_pred_i)
                volumes.append(vol_i)

            # ED = max volume, ES = min volume
            edv = max(volumes)
            esv = min(volumes)
            ef_pred = ((edv - esv) / edv)*100 if edv > 1e-6 else 0.0

            # Get ground-truth EF from test_df
            this_file = file_names[0]
            row = test_df[test_df["FileName"] == this_file]
            if len(row) == 0:
                # No match => skip
                continue
            gt_ef = float(row.iloc[0]["EF"])

            # Store them:
            efs_pred.append(ef_pred)
            efs_true.append(gt_ef)

            total_samples += batch_size

    # ---------------------------
    # 2) Segmentation metrics
    # ---------------------------
    test_loss /= total_samples
    test_dice /= total_samples
    print(f"Test Loss (segmentation): {test_loss}")
    print(f"Test Dice (segmentation): {test_dice}")

    # ---------------------------
    # 3) EF regression metrics
    # ---------------------------
    efs_pred = np.array(efs_pred)
    efs_true = np.array(efs_true)

    if len(efs_true) > 0:
        # MAE
        mae_ef = np.mean(np.abs(efs_pred - efs_true))

        # MAPE => mean absolute percentage error
        # for each sample, 100*|pred - true|/true, then average
        mape_ef = np.mean(np.abs(efs_pred - efs_true) / np.clip(efs_true, 1e-6, None)) * 100.0

        print(f"EF Mean Absolute Error (MAE): {mae_ef}")
        print(f"EF Mean Absolute Percentage Error (MAPE): {mape_ef}%")
    else:
        print("No EF data to compute regression metrics.")

def visualize_test_predictions(model,test_loader,device):
    model.eval()
    num_batches_to_show = 5
    total_batches = len(test_loader)

    # Randomly choose 5 unique batch indices
    batch_indices = random.sample(range(total_batches), num_batches_to_show)

    # Convert test_loader to list for random access
    test_batches = list(test_loader)

    with torch.no_grad():
        for batch_count, batch_idx in enumerate(batch_indices, 1):
            images, masks, file_names = test_batches[batch_idx]
            images = images.to(device)
            masks = masks.to(device)

            print(f"\n🔹 Showing Batch {batch_count} (Index: {batch_idx})\n")

            # Model prediction
            outputs = model(images)
            preds = (outputs > 0.5).float()

            # Show up to 4 samples from batch
            num_to_show = min(4, images.size(0))
            for i in range(num_to_show):
                img_np = TF.to_pil_image(images[i].cpu())
                mask_np = TF.to_pil_image(masks[i].cpu().squeeze(0))
                pred_np = TF.to_pil_image(preds[i].cpu().squeeze(0))

                plt.figure(figsize=(10, 3))

                plt.subplot(1, 3, 1)
                plt.title("Original Image")
                plt.imshow(img_np)

                plt.subplot(1, 3, 2)
                plt.title("Ground Truth Mask")
                plt.imshow(img_np)
                plt.imshow(mask_np, alpha=0.4, cmap='Reds')

                plt.subplot(1, 3, 3)
                plt.title("Predicted Mask")
                plt.imshow(img_np)
                plt.imshow(pred_np, alpha=0.4, cmap='Reds')

                plt.tight_layout()
                plt.show()