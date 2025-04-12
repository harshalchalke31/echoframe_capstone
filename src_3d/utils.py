from torch.utils.data import DataLoader
import random
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F
import imageio
import torchvision.transforms as T
import math
import io
from PIL import Image
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import pandas as pd
import cv2
from torch.profiler import profile, record_function, ProfilerActivity
from src.utils import BCEDiceLoss

def visualize_random_video_from_loader(loader, max_batches=5):
    """
    Visualizes a random sample from a random batch from a loader.
    Assumes output is (videos, masks) with shapes:
    videos: [B, 3, D, H, W]
    masks:  [B, 1, H, W]
    """
    batch_idx = random.randint(0, max_batches - 1)

    for i, (videos, masks) in enumerate(loader):
        if i != batch_idx:
            continue

        sample_idx = random.randint(0, videos.shape[0] - 1)
        video = videos[sample_idx]  # [3, D, H, W]
        mask = masks[sample_idx][0]  # [H, W]

        video = video.permute(1, 2, 3, 0)  # [D, H, W, C]
        D = video.shape[0]
        grid_size = int(np.ceil(np.sqrt(D)))

        fig, axs = plt.subplots(grid_size, grid_size, figsize=(12, 12))
        fig.suptitle(f"Batch {batch_idx}, Sample {sample_idx}", fontsize=16)

        for i in range(grid_size * grid_size):
            row, col = divmod(i, grid_size)
            ax = axs[row][col]

            if i < D:
                frame = video[i].numpy()
                ax.imshow(frame)
                if i == D // 2:
                    ax.imshow(mask.numpy(), cmap='jet', alpha=0.4)
                    ax.set_title("Center Frame + Mask", fontsize=10)
                else:
                    ax.set_title(f"Frame {i}", fontsize=8)
            ax.axis("off")

        plt.tight_layout()
        plt.show()
        break




def dice_coefficient(preds, targets, smooth=1e-5, threshold=0.5):
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()
    preds = preds.view(-1)
    targets = targets.view(-1)
    intersection = (preds * targets).sum()
    return (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)

def test_3d_unet(model, test_loader, model_path, device='cuda'):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    total_dice, total_samples = 0.0, 0

    with torch.no_grad():
        for images, masks,_ in tqdm(test_loader, desc="Testing"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            center = images.shape[2] // 2
            preds = outputs[:, :, center]
            batch_dice = dice_coefficient(preds, masks)
            total_dice += batch_dice.item() * images.size(0)
            total_samples += images.size(0)

    print(f"Test Dice Coefficient (center frame only): {total_dice / total_samples:.4f}")


def visualize_clip_with_overlay(model, test_loader, model_path, device='cuda', sample_idx=None):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Get a batch
    batch = next(iter(test_loader))
    images, masks,_ = batch
    images = images.to(device)

    with torch.no_grad():
        outputs = model(images)

    # Choose a sample
    idx = sample_idx if sample_idx is not None else 0
    input_clip = images[idx].cpu().permute(1, 2, 3, 0)     # [T, H, W, C]
    pred_clip = torch.sigmoid(outputs[idx][0]).cpu()       # [T, H, W]

    T = input_clip.shape[0]
    cols = 4
    rows = math.ceil(T / cols)

    fig, axs = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axs = axs.flatten()

    for t in range(T):
        axs[t].imshow(input_clip[t].numpy())
        axs[t].imshow(pred_clip[t].numpy(), alpha=0.5, cmap='Reds')
        axs[t].set_title(f"t={t}")
        axs[t].axis("off")

    # Hide unused axes if any
    for t in range(T, len(axs)):
        axs[t].axis("off")

    plt.tight_layout()
    plt.show()



def save_overlay_gif_from_loader(model, test_loader, model_path, save_path="output_clip.gif",
                                 batch_idx=0, sample_idx=0, device="cuda", fps=5):
    # Load model weights and prepare evaluation
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Get the specified batch from test_loader
    for i, (images, masks,_) in enumerate(test_loader):
        if i == batch_idx:
            images = images.to(device)
            with torch.no_grad():
                outputs = model(images)
            break
    else:
        raise ValueError(f"Batch index {batch_idx} out of range!")

    # Choose sample from the batch
    input_clip = images[sample_idx].cpu().permute(1, 2, 3, 0)  # [T, H, W, C]
    pred_clip = torch.sigmoid(outputs[sample_idx][0]).cpu()    # [T, H, W]

    frames = []
    # Iterate over every frame in the clip
    for t in range(input_clip.shape[0]):
        frame = input_clip[t].numpy()
        mask = pred_clip[t].numpy()

        fig, ax = plt.subplots(figsize=(3, 3))
        # Bind the canvas to the figure explicitly
        FigureCanvas(fig)
        ax.imshow(frame)
        ax.imshow(mask, alpha=0.5, cmap='Reds')
        ax.axis('off')

        # Save the figure to an in-memory buffer
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        img = Image.open(buf)
        frames.append(np.array(img))
        buf.close()
        plt.close(fig)

    # Save frames as a GIF
    imageio.mimsave(save_path, frames)
    print(f"✅ Saved overlay GIF to: {save_path}")



#---------------------------------Testing code---------------------------------------#
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
    d12, d23, d13 = np.linalg.norm(p1 - p2), np.linalg.norm(p2 - p3), np.linalg.norm(p1 - p3)
    if d12 < d23 and d12 < d13:
        return p3, p1, p2
    elif d23 < d12 and d23 < d13:
        return p1, p3, p2
    else:
        return p2, p1, p3


def simpsons_biplane(cnt, frame):
    _, tri = cv2.minEnclosingTriangle(cnt)
    tri = tri.reshape(-1, 2)
    pts = [cnt[closest_point(t, cnt[:, 0, :])[0], 0, :] for t in tri]
    apex, mv1, mv2 = label_points(*pts)
    mid_mv = (mv1 + mv2) / 2.0
    slope = apex - mid_mv
    slope_len = np.linalg.norm(slope)
    if slope_len < 1e-6: return 0.0
    slope /= slope_len

    h, w = frame.shape[:2]
    diag = int(np.sqrt(h**2 + w**2))
    start = (apex - slope * diag * 2).astype(int)
    end = (mid_mv + slope * diag * 2).astype(int)

    blank = np.zeros_like(frame, dtype=np.uint8)
    mask = cv2.drawContours(blank.copy(), [cnt], 0, 1, 1)
    width = 2
    mv_diam = np.linalg.norm(mv1 - mv2)

    while True:
        line = cv2.line(blank.copy(), tuple(start), tuple(end), 1, width)
        intersect = np.logical_and(mask, line)
        pts = np.column_stack(np.where(intersect == 1))
        if pts.size == 0:
            width += 1
            if width > diag * 2: return 0.0
            continue
        idx, dists = farthest_point(np.flip(apex), pts)
        if len(pts) >= 2 and dists[idx] >= 0.9 * mv_diam:
            break
        width += 1
        if width > diag * 2: return 0.0

    intercept = np.flip(pts[idx])
    lv_len_cm = np.linalg.norm(apex - intercept) * PX_TO_CM
    area_cm2 = cv2.contourArea(cnt) * (PX_TO_CM ** 2)
    return (8.0 * area_cm2**2) / (3.0 * np.pi * lv_len_cm) if lv_len_cm > 1e-6 else 0.0


def compute_volume_from_mask(mask):
    mask = (mask * 255).astype(np.uint8)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts: return 0.0
    max_cnt = max(cnts, key=cv2.contourArea)
    return simpsons_biplane(max_cnt, np.zeros_like(mask))


def dice_coefficient(preds, targets, smooth=1e-5, threshold=0.5):
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()
    preds = preds.view(-1)
    targets = targets.view(-1)
    intersection = (preds * targets).sum()
    return (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)




def find_local_extrema(volumes):
    """
    Find local maxima and minima in a 1D volume array
    by looking at sign changes in the slope.
    Returns indices of maxima and indices of minima.
    """
    maxima_idx, minima_idx = [], []
    for i in range(1, len(volumes) - 1):
        # Compare neighboring frames
        if volumes[i] > volumes[i - 1] and volumes[i] > volumes[i + 1]:
            maxima_idx.append(i)
        if volumes[i] < volumes[i - 1] and volumes[i] < volumes[i + 1]:
            minima_idx.append(i)
    return maxima_idx, minima_idx


def compute_mean_ef(volumes):
    """
    Given a list/array of volumes across frames, find all local maxima and minima,
    pair each max with the subsequent min, and compute EF for each pair.
    Then return the average EF across all cycles found.
    """
    if len(volumes) < 3:
        # Not enough frames to compute local maxima/minima
        return 0.0

    maxima_idx, minima_idx = find_local_extrema(volumes)
    if not maxima_idx or not minima_idx:
        # Fallback: no local max/min found
        return 0.0

    efs = []
    min_j = 0
    # For each max, find the next minima that occurs after that max
    for max_i in maxima_idx:
        while min_j < len(minima_idx) and minima_idx[min_j] < max_i:
            min_j += 1
        if min_j >= len(minima_idx):
            break
        ed_vol = volumes[max_i]   # "ED" volume
        es_vol = volumes[minima_idx[min_j]]  # "ES" volume
        if ed_vol > 1e-6:
            ef_value = (ed_vol - es_vol) / ed_vol * 100.0
            efs.append(ef_value)

    if len(efs) == 0:
        return 0.0
    return float(np.mean(efs))


def test_3d_model_full(model, test_loader, test_df, model_path,
                       device='cuda', save_path="./logs/test_3d_metrics_model1.csv"):
    """
    Upgraded test function to:
    - Predict on every frame in each 3D clip
    - Compute volume at all frames
    - Detect multiple ED/ES pairs, compute EF for each, and average
    - Log volume series in CSV
    """
    # 1) Load model weights and set eval mode
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    criterion = BCEDiceLoss()

    results = []
    efs_pred, efs_true = [], []
    total_loss, total_dice = 0.0, 0.0

    # 2) Loop through testing data
    for images, masks, file_names in tqdm(test_loader, desc="Testing"):
        images, masks = images.to(device), masks.to(device)
        bs, _, T, H, W = images.shape

        # 3) Profile for GFLOPs and memory usage
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     record_shapes=True, profile_memory=True, with_flops=True) as prof:
            with torch.no_grad(), record_function("model_inference"):
                outputs = model(images)  # shape: [B, 1, T, H, W]

        # 4) Compute segmentation loss & Dice for the *center* frame (for reference)
        center_frame = T // 2
        preds_center = outputs[:, :, center_frame]  # shape [B, 1, H, W]
        loss = criterion(preds_center, masks)       # only comparing to the center mask
        dice = dice_coefficient(preds_center, masks)

        total_loss += loss.item() * bs
        total_dice += dice.item() * bs

        # 5) Threshold outputs for all frames
        preds_sigmoid = torch.sigmoid(outputs)       # [B, 1, T, H, W]
        preds_bin = (preds_sigmoid.cpu().numpy() > 0.5).astype(np.uint8)

        # 6) For each clip in the batch, compute volumes at each frame
        for i in range(bs):
            volumes = []
            for t in range(T):
                # shape of preds_bin[i, 0, t] is (H, W)
                frame_mask = preds_bin[i, 0, t]
                vol = compute_volume_from_mask(frame_mask)  # user-provided function
                volumes.append(vol)

            # 7) Find multiple ED/ES pairs, compute EF for each, then average
            mean_ef = compute_mean_ef(volumes)

            # 8) Gather info
            fname = file_names[i]
            gt_row = test_df[test_df['FileName'] == fname]
            gt_ef = float(gt_row['EF'].values[0]) if not gt_row.empty else 0.0

            # 9) Profile stats
            flops = prof.key_averages().total_average().flops / 1e9  # GFLOPs
            mem = prof.key_averages().total_average().self_device_memory_usage / 1e6

            # For global metrics
            efs_pred.append(mean_ef)
            efs_true.append(gt_ef)

            # 10) Store results (include entire volume series for later plotting)
            results.append({
                'FileName': fname,
                'Volumes': ";".join(map(str, volumes)),  # store as semicolon-delimited
                'MeanEF_Pred': mean_ef,
                'EF_True': gt_ef,
                'GFLOPs': flops,
                'Memory_MB': mem,
                'Dice_CenterFrame': dice.item(),
                'Loss': loss.item()
            })

    # 11) Convert to DataFrame & save
    df_results = pd.DataFrame(results)
    df_results.to_csv(save_path, index=False)

    # 12) Print metrics
    n_samples = len(test_loader.dataset)
    avg_loss = total_loss / n_samples
    avg_dice = total_dice / n_samples
    mae = np.mean(np.abs(np.array(efs_pred) - np.array(efs_true)))
    mape = np.mean(np.abs(np.array(efs_pred) - np.array(efs_true))
                   / np.clip(np.array(efs_true), 1e-6, None)) * 100.0

    print(f"\n🧪 Segmentation Dice (center frame only): {avg_dice:.4f}")
    print(f"📉 Segmentation Loss (center frame):       {avg_loss:.4f}")
    print(f"📈 EF MAE:                                 {mae:.2f}")
    print(f"📊 EF MAPE:                                {mape:.2f} %")



