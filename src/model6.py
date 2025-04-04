import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class PretrainedUNETR(nn.Module):
    def __init__(
        self,
        num_classes: int = 1,
        img_size: int = 112,               
        patch_size: int = 16,
        encoder_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        skip_indices: tuple = (3, 6, 9),
    ):
        """
        Args:
            num_classes: Number of segmentation classes (or 1 for binary seg).
            img_size: Input image resolution = 112.
            patch_size: Patch size used in the transformer backbone.
            encoder_name: Name of the pretrained ViT backbone from timm.
            pretrained: Whether to load pretrained weights.
            skip_indices: Which transformer blocks (1-indexed) to use as skip connections.
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # 1) Create the ViT backbone using timm
        self.vit = timm.create_model(
            encoder_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool='',
            img_size=img_size
        )
        # 2) If timm complains about image size not matching 224 - disable strict checking:
        self.vit.patch_embed.strict_img_size = False

        self.hidden_dim = self.vit.embed_dim
        self.use_cls_token = hasattr(self.vit, "cls_token")

        # Convert provided skip indices (1-indexed) to 0-indexed
        self.skip_indices = [i - 1 for i in skip_indices]
        self.num_blocks = len(self.vit.blocks)

        # ======================
        # Decoder (U-Net style)
        # ======================
        #  -- ConvTranspose2d for each decoder stage --
        # Because the final patch-grid for a 112×112 input is 7×7 (112/16),
        # these deconvs are sized to progressively upsample back to 112×112.

        # Block 1: Upsample from the deepest skip (final transformer block).
        self.dec1_deconv = nn.ConvTranspose2d(
            in_channels=self.hidden_dim, out_channels=256,
            kernel_size=2, stride=2
        )
        # Block 2: Fuse with next skip
        self.dec2_deconv = nn.ConvTranspose2d(
            in_channels=256, out_channels=128,
            kernel_size=2, stride=2
        )
        self.dec2_skip = nn.ConvTranspose2d(
            in_channels=self.hidden_dim, out_channels=128,
            kernel_size=4, stride=4
        )
        self.dec2_conv = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        # Block 3: Fuse with shallower skip
        self.dec3_deconv = nn.ConvTranspose2d(
            in_channels=128, out_channels=64,
            kernel_size=2, stride=2
        )
        self.dec3_skip = nn.ConvTranspose2d(
            in_channels=self.hidden_dim, out_channels=64,
            kernel_size=8, stride=8
        )
        self.dec3_conv = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # Block 4: Fuse with original input
        self.dec4_deconv = nn.ConvTranspose2d(
            in_channels=64, out_channels=32,
            kernel_size=2, stride=2
        )
        # We do a small conv on the original input to use it as a "skip"
        self.dec4_skip = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.dec4_conv = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        # final segmentation head
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        B, _, H, W = x.shape
        x_orig = x  # Save original input for the last decoder block

        # ===== Encoder =====
        # 1) Get patch embeddings
        x_patch = self.vit.patch_embed(x)
        if x_patch.dim() == 4:
            # shape = (B, hidden_dim, H_patch, W_patch)
            H_patch, W_patch = x_patch.shape[2], x_patch.shape[3]
            x_patch = x_patch.flatten(2).transpose(1, 2)  # -> (B, num_patches, hidden_dim)
        elif x_patch.dim() == 3:
            # shape = (B, num_patches, hidden_dim); need to compute H_patch, W_patch
            B, num_patches, _ = x_patch.shape
            H_patch = W_patch = int(math.sqrt(num_patches))
        else:
            raise ValueError("Unexpected patch_embed output dimension.")

        # 2) Add positional embeddings
        if self.use_cls_token:
            # the first position is the CLS token embed, so skipping that
            x_patch = x_patch + self.vit.pos_embed[:, 1:, :]
        else:
            x_patch = x_patch + self.vit.pos_embed

        # 3) Pass through transformer blocks, gather skip features
        skip_tokens = []
        for i, block in enumerate(self.vit.blocks):
            x_patch = block(x_patch)
            if i in self.skip_indices:
                skip_tokens.append(x_patch)
        skip_tokens = skip_tokens[::-1]  # reverse order (deepest skip first)

        # 4) Reshape transformer outputs back to 2D feature maps
        x_enc = x_patch.transpose(1, 2).view(B, self.hidden_dim, H_patch, W_patch)
        skip0 = skip_tokens[0].transpose(1, 2).view(B, self.hidden_dim, H_patch, W_patch) if len(skip_tokens) > 0 else None
        skip1 = skip_tokens[1].transpose(1, 2).view(B, self.hidden_dim, H_patch, W_patch) if len(skip_tokens) > 1 else None
        skip2 = skip_tokens[2].transpose(1, 2).view(B, self.hidden_dim, H_patch, W_patch) if len(skip_tokens) > 2 else None

        # ===== Decoder =====
        # Block 1: Upsample the deepest skip
        x_dec1 = self.dec1_deconv(skip0) if skip0 is not None else x_enc

        # Block 2: Fuse with skip1
        if skip1 is not None:
            x_dec2_main = self.dec2_deconv(x_dec1)
            x_dec2_skip = self.dec2_skip(skip1)
            x_dec2 = torch.cat([x_dec2_main, x_dec2_skip], dim=1)
            x_dec2 = self.dec2_conv(x_dec2)
        else:
            x_dec2 = self.dec2_deconv(x_dec1)

        # Block 3: Fuse with skip2
        if skip2 is not None:
            x_dec3_main = self.dec3_deconv(x_dec2)
            x_dec3_skip = self.dec3_skip(skip2)
            x_dec3 = torch.cat([x_dec3_main, x_dec3_skip], dim=1)
            x_dec3 = self.dec3_conv(x_dec3)
        else:
            x_dec3 = self.dec3_deconv(x_dec2)

        # Block 4: Fuse with original input
        x_dec4_main = self.dec4_deconv(x_dec3)
        # interpolate original input to match x_dec4_main shape
        x_orig_down = F.interpolate(x_orig, size=x_dec4_main.shape[2:], mode='bilinear', align_corners=False)
        x_dec4_skip = self.dec4_skip(x_orig_down)
        x_dec4 = torch.cat([x_dec4_main, x_dec4_skip], dim=1)
        x_dec4 = self.dec4_conv(x_dec4)

        out = self.final_conv(x_dec4)
        return out


if __name__ == "__main__":
    model = PretrainedUNETR(
        num_classes=1,
        img_size=112,                  
        patch_size=16,
        encoder_name="vit_base_patch16_224",
        pretrained=True,
        skip_indices=(3, 6, 9)          
    )
    dummy_input = torch.randn(2, 3, 112, 112)
    output = model(dummy_input)
    print("Output shape:", output.shape)
