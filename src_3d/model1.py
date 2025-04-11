import torch
import torch.nn as nn
import torch.nn.functional as F


# Normalization helper: can switch easily to GroupNorm, etc.
def get_norm_3d(num_channels, norm_type="bn", num_groups=4):
    """
    Returns 3D normalization layer: either BatchNorm3d or GroupNorm, etc.
    Modify as needed.
    """
    if norm_type == "bn":
        return nn.BatchNorm3d(num_channels)
    elif norm_type == "gn":
        return nn.GroupNorm(num_groups, num_channels)
    else:
        raise ValueError(f"Unknown norm_type={norm_type}")



# 1) Squeeze-and-Excitation (3D) with 1x1x1 conv

class SELayer3D(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Conv3d(channel, channel // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(channel // reduction, channel, kernel_size=1, bias=False),
            nn.Hardsigmoid(inplace=True),
        )

    def forward(self, x):
        b, c, d, h, w = x.size()
        # Squeeze
        y = self.avg_pool(x)
        # Excitation
        y = self.fc(y)
        return x * y



# 2) Inverted Residual Block (3D) for MobileNetV3 encoder

class InvertedResidual3D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,               # (1,1,1) or (1,2,2) etc.
        expand_ratio,
        use_se,
        activation,
        norm_type="bn"
    ):
        super().__init__()
        hidden_dim = int(round(in_channels * expand_ratio))
        self.use_res_connect = (stride == (1,1,1) and in_channels == out_channels)

        # Choose activation
        if activation == "RE":
            act_layer = nn.ReLU(inplace=True)
        elif activation == "HS":
            act_layer = nn.Hardswish(inplace=True)
        else:
            raise NotImplementedError(f"Unknown activation {activation}")

        layers = []
        # (1) Pointwise expansion
        if expand_ratio != 1:
            layers.append(nn.Conv3d(in_channels, hidden_dim, 1, bias=False))
            layers.append(get_norm_3d(hidden_dim, norm_type))
            layers.append(act_layer)

        # (2) Depthwise
        padding = kernel_size // 2  # assumes odd kernel size
        layers.append(
            nn.Conv3d(
                hidden_dim,
                hidden_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=hidden_dim,
                bias=False
            )
        )
        layers.append(get_norm_3d(hidden_dim, norm_type))
        if use_se:
            layers.append(SELayer3D(hidden_dim))
        layers.append(act_layer)

        # (3) Pointwise projection
        layers.append(nn.Conv3d(hidden_dim, out_channels, 1, bias=False))
        layers.append(get_norm_3d(out_channels, norm_type))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)



# 3) MobileNetV3-Large 3D Encoder

class MobileNetV3Large3DEncoder(nn.Module):
    """
    Stages:
      feats[0]:  input itself (no stride)
      feats[1]:  stem => [16] stride=(1,1,1)
      feats[2] & feats[3]: 1st down => [24]
      feats[4] & feats[5]: 2nd down => [40]
      feats[6] & feats[7]: 3rd down => [80]
      feats[8] & feats[9]: 4th down => [112]
      feats[10] & feats[11]: final => [160]

    set skip_indices = [1, 3, 5, 7, 9].
    The final output is feats[11].
    """
    def __init__(self, in_channels=3, norm_type="bn"):
        super().__init__()
        # Store skip indices (for a UNet)
        self.skip_indices = [1, 3, 5, 7, 9]
        self.final_idx = 11

        # (1) Stem => out=16, stride=(1,1,1)
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=3,
                      stride=(1,1,1), padding=1, bias=False),
            get_norm_3d(16, norm_type),
            nn.Hardswish(inplace=True)
        )

        # Down 1 => out=24
        self.block2 = InvertedResidual3D(16, 24, 3, (1,2,2),
                                         expand_ratio=4, use_se=False, activation="RE", norm_type=norm_type)
        self.block3 = InvertedResidual3D(24, 24, 3, (1,1,1),
                                         expand_ratio=3, use_se=False, activation="RE", norm_type=norm_type)

        # Down 2 => out=40
        self.block4 = InvertedResidual3D(24, 40, 5, (1,2,2),
                                         expand_ratio=3, use_se=True, activation="HS", norm_type=norm_type)
        self.block5 = InvertedResidual3D(40, 40, 5, (1,1,1),
                                         expand_ratio=3, use_se=True, activation="HS", norm_type=norm_type)

        # Down 3 => out=80
        self.block6 = InvertedResidual3D(40, 80, 3, (1,2,2),
                                         expand_ratio=6, use_se=False, activation="HS", norm_type=norm_type)
        self.block7 = InvertedResidual3D(80, 80, 3, (1,1,1),
                                         expand_ratio=2.5, use_se=False, activation="HS", norm_type=norm_type)

        # Down 4 => out=112
        self.block8 = InvertedResidual3D(80, 112, 3, (1,2,2),
                                         expand_ratio=6, use_se=True, activation="HS", norm_type=norm_type)
        self.block9 = InvertedResidual3D(112, 112, 3, (1,1,1),
                                         expand_ratio=6, use_se=True, activation="HS", norm_type=norm_type)

        # Final => out=160
        self.block10 = InvertedResidual3D(112, 160, 5, (1,1,1),
                                          expand_ratio=6, use_se=True, activation="HS", norm_type=norm_type)
        self.block11 = InvertedResidual3D(160, 160, 5, (1,1,1),
                                          expand_ratio=6, use_se=True, activation="HS", norm_type=norm_type)

    def forward(self, x):
        # feats[0] = the original input
        feats = [x]

        # Stem => feats[1]
        x = self.stem(x)
        feats.append(x)

        # Down 1 => feats[2], feats[3]
        x = self.block2(x)
        feats.append(x)
        x = self.block3(x)
        feats.append(x)

        # Down 2 => feats[4], feats[5]
        x = self.block4(x)
        feats.append(x)
        x = self.block5(x)
        feats.append(x)

        # Down 3 => feats[6], feats[7]
        x = self.block6(x)
        feats.append(x)
        x = self.block7(x)
        feats.append(x)

        # Down 4 => feats[8], feats[9]
        x = self.block8(x)
        feats.append(x)
        x = self.block9(x)
        feats.append(x)

        # Final => feats[10], feats[11]
        x = self.block10(x)
        feats.append(x)
        x = self.block11(x)
        feats.append(x)

        return feats



# A 3D residual block for the decoder

class ResidualBlock3D(nn.Module):
    """
    A basic 3D residual block with two 3x3x3 conv layers.
    """
    def __init__(self, channels, dropout=0.0, norm_type="bn"):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = get_norm_3d(channels, norm_type)
        self.relu1 = nn.ReLU(inplace=True)
        self.drop1 = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()

        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = get_norm_3d(channels, norm_type)

        self.relu2 = nn.ReLU(inplace=True)
        self.drop2 = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu1(out)
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.norm2(out)
        # Residual Add
        out = out + identity

        out = self.relu2(out)
        out = self.drop2(out)
        return out



# 4) DecoderBlock3D with residual blocks

class DecoderBlock3D(nn.Module):
    def __init__(
        self,
        in_c,
        out_c,
        skip_c,
        dropout=0.0,
        norm_type="bn",
        num_residual_blocks=2
    ):
        """
        A deeper decoder stage:
          1) Upsample
          2) Concatenate skip
          3) A 3x3 conv (optional)
          4) N residual blocks (each with skip conn)
        """
        super().__init__()
        self.up = nn.ConvTranspose3d(
            in_channels=in_c,
            out_channels=out_c,
            kernel_size=(1, 2, 2),
            stride=(1, 2, 2)
        )

        # Optional 3x3 conv to merge skip
        self.merge_conv = nn.Sequential(
            nn.Conv3d(out_c + skip_c, out_c, kernel_size=3, padding=1, bias=False),
            get_norm_3d(out_c, norm_type),
            nn.ReLU(inplace=True),
        )

        # Stack multiple residual blocks for "deeper" decoding
        self.resblocks = nn.Sequential(
            *[ResidualBlock3D(out_c, dropout=dropout, norm_type=norm_type)
              for _ in range(num_residual_blocks)]
        )

    def forward(self, x, skip):
        # Upsample
        x = self.up(x)

        # Fix shape mismatch (rounding from stride etc.)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)

        # Concat skip
        x = torch.cat([skip, x], dim=1)

        # Merge channels
        x = self.merge_conv(x)

        # Apply deeper residual blocks
        x = self.resblocks(x)
        return x



# 5) MobileNetV3-Large UNet 3D
class MobileNetV3UNet3D(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=1,
        dropout=0.0,
        norm_type="bn",
        final_activation=None,  # 'sigmoid', 'softmax', or None
        num_res_blocks=2        # number of residual blocks in each decoder stage
    ):
        super().__init__()
        # Create the encoder
        self.encoder = MobileNetV3Large3DEncoder(in_channels, norm_type=norm_type)
        # For convenience
        self.skip_ids = self.encoder.skip_indices  # [1,3,5,7,9]
        self.bottleneck_id = self.encoder.final_idx  # 11

        # The channels at each skip stage (derived from standard config of MobileNetV3)
        # e.g. feats[1]->16, feats[3]->24, feats[5]->40, feats[7]->80, feats[9]->112, feats[11]->160
        self.skip_channels = {
            1: 16,
            3: 24,
            5: 40,
            7: 80,
            9: 112
        }
        bottleneck_channels = 160

        # Decoder blocks
        self.dec4 = DecoderBlock3D(
            in_c=bottleneck_channels,
            out_c=112,
            skip_c=self.skip_channels[9],
            dropout=dropout,
            norm_type=norm_type,
            num_residual_blocks=num_res_blocks
        )
        self.dec3 = DecoderBlock3D(
            in_c=112,
            out_c=80,
            skip_c=self.skip_channels[7],
            dropout=dropout,
            norm_type=norm_type,
            num_residual_blocks=num_res_blocks
        )
        self.dec2 = DecoderBlock3D(
            in_c=80,
            out_c=40,
            skip_c=self.skip_channels[5],
            dropout=dropout,
            norm_type=norm_type,
            num_residual_blocks=num_res_blocks
        )
        self.dec1 = DecoderBlock3D(
            in_c=40,
            out_c=24,
            skip_c=self.skip_channels[3],
            dropout=dropout,
            norm_type=norm_type,
            num_residual_blocks=num_res_blocks
        )

        # final_up => merges with feats[1] => 16
        self.final_up = nn.ConvTranspose3d(24, 16, (1,2,2), stride=(1,2,2))
        self.final_merge = nn.Sequential(
            nn.Conv3d(16 + 16, 16, kernel_size=3, padding=1, bias=False),
            get_norm_3d(16, norm_type),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=dropout) if dropout > 0.0 else nn.Identity(),
        )
        self.final_conv = nn.Conv3d(16, out_channels, kernel_size=1)

        self.final_activation = final_activation

    def forward(self, x):
        """
        Forward pass. Returns final logits or activated output if final_activation is set.
        """
        feats = self.encoder(x)

        # Bottleneck
        out = feats[self.bottleneck_id]  # feats[11] => [160, ...]
        # dec4 => skip with feats[9] => [112]
        out = self.dec4(out, feats[self.skip_ids[4]])  # feats[9]
        # dec3 => skip with feats[7] => [80]
        out = self.dec3(out, feats[self.skip_ids[3]])  # feats[7]
        # dec2 => skip with feats[5] => [40]
        out = self.dec2(out, feats[self.skip_ids[2]])  # feats[5]
        # dec1 => skip with feats[3] => [24]
        out = self.dec1(out, feats[self.skip_ids[1]])  # feats[3]

        # final up => skip feats[1] => [16]
        out = self.final_up(out)
        if out.shape[2:] != feats[self.skip_ids[0]].shape[2:]:
            out = F.interpolate(out, size=feats[self.skip_ids[0]].shape[2:], 
                                mode='trilinear', align_corners=False)
        out = torch.cat([feats[self.skip_ids[0]], out], dim=1)

        out = self.final_merge(out)
        out = self.final_conv(out)

        # Optional final activation
        if self.final_activation == "sigmoid":
            out = torch.sigmoid(out)
        elif self.final_activation == "softmax":
            # typically for multi-class => out.shape is [B, C, T, H, W]
            out = F.softmax(out, dim=1)

        return out


if __name__ == "__main__":
    model = MobileNetV3UNet3D(
        in_channels=3,
        out_channels=2,      
        dropout=0.2,         
        norm_type="bn",      
        final_activation="softmax", 
        num_res_blocks=2     
    ).cuda()

    # Random temporal data [B, C, T, H, W]
    dummy_input = torch.randn(2, 3, 16, 112, 112).cuda()

    model.train()
    out = model(dummy_input)
    print("Train mode output shape:", out.shape)

    model.eval()
    # Force dropout submodules to remain active for MC test
    for m in model.modules():
        if isinstance(m, nn.Dropout3d):
            m.train()

    # Forward pass #1
    out1 = model(dummy_input)
    # Forward pass #2
    out2 = model(dummy_input)
    # out1 and out2 will differ due to dropout if p>0

    print("Eval mode + MC dropout output shapes:", out1.shape, out2.shape)
