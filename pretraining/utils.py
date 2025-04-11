import random
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch
from torch.nn import functional as F
from kornia.metrics import ssim


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



#----------------------------Metrics------------------------------------------#
def psnr(pred, target, max_val=1.0):
    mse = F.mse_loss(pred, target)
    return 20 * torch.log10(max_val / torch.sqrt(mse))


def ssim_score(pred, target):
    # Convert [B, C, T, H, W] → [B*T, C, H, W]
    B, C, T, H, W = pred.shape
    pred = pred.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
    target = target.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
    return ssim(pred, target, window_size=11).mean()


def temporal_consistency_loss(output, target):
    # delta = frame_t+1 - frame_t → compare temporal derivatives
    delta_out = output[:, :, 1:] - output[:, :, :-1]
    delta_gt = target[:, :, 1:] - target[:, :, :-1]
    return F.mse_loss(delta_out, delta_gt)

#-----------------------------Test------------------------------------------------#

def test_autoencoder(model, test_loader, model_path, device='cuda'):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    total_psnr = total_ssim = total_tdc = total_samples = 0

    with torch.no_grad():
        for images in test_loader:
            images = images.to(device)
            outputs = model(images)

            batch_size = images.size(0)
            total_samples += batch_size
            total_psnr += psnr(outputs, images).item() * batch_size
            total_ssim += ssim_score(outputs, images).item() * batch_size
            total_tdc += temporal_consistency_loss(outputs, images).item() * batch_size

    print(f"PSNR: {total_psnr / total_samples:.2f} dB")
    print(f"SSIM: {total_ssim / total_samples:.4f}")
    print(f"TDC:  {total_tdc / total_samples:.6f}")


def visualize_random_batch(model, test_loader, model_path, device='cuda'):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    batch = next(iter(test_loader))
    inputs = batch.to(device)

    with torch.no_grad():
        outputs = model(inputs)

    idx = random.randint(0, inputs.size(0) - 1)
    t = random.randint(0, inputs.size(2) - 1)

    input_frame = inputs[idx, :, t].cpu().permute(1, 2, 0).numpy()
    output_frame = outputs[idx, :, t].cpu().permute(1, 2, 0).numpy()

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(input_frame)
    axs[0].set_title("Original Frame")
    axs[0].axis("off")

    axs[1].imshow(output_frame)
    axs[1].set_title("Reconstructed Frame")
    axs[1].axis("off")
    plt.tight_layout()
    plt.show()

