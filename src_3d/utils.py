from torch.utils.data import DataLoader
import random
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import numpy as np

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