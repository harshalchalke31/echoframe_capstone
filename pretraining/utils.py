import random
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch


def visualize_random_video_from_loader(loader, max_batches=5):
    """
    Visualizes a random sample from a random batch from a loader.
    Assumes output is (videos, masks) with shapes:
    videos: [B, 3, D, H, W]
    """
    batch_idx = random.randint(0, max_batches - 1)

    for i, videos in enumerate(loader):
        if i != batch_idx:
            continue

        sample_idx = random.randint(0, videos.shape[0] - 1)
        video = videos[sample_idx]  # [3, D, H, W]

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
                ax.set_title(f"Frame {i}", fontsize=8)
            ax.axis("off")

        plt.tight_layout()
        plt.show()
        break


#------------------------Custom Loss-----------------------------------#
class MaskedMSELoss(nn.Module):
    def forward(self, input, target, mask):
        # input, target: [B, C, T, H, W]
        # mask: [B, 1, T, H, W] with 1s for visible (unmasked), 0s for masked
        diff = (input - target) ** 2
        masked_diff = diff * mask
        return masked_diff.sum() / mask.sum()
    
def generate_random_mask(shape, mask_ratio=0.75, device='cpu'):
    """
    Generate binary mask of shape [B, 1, T, H, W]
    1 = visible, 0 = masked (reconstruction target)
    """
    B, _, T, H, W = shape
    total_elements = T * H * W
    num_visible = int((1 - mask_ratio) * total_elements)

    mask = torch.zeros(B, T * H * W, device=device)
    for i in range(B):
        visible_indices = torch.randperm(T * H * W)[:num_visible]
        mask[i, visible_indices] = 1.0

    mask = mask.view(B, 1, T, H, W)
    return mask



