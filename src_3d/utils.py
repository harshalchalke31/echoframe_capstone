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
        for images, masks in tqdm(test_loader, desc="Testing"):
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
    images, masks = batch
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
    for i, (images, masks) in enumerate(test_loader):
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



