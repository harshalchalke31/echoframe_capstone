import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

# -----------------------------------------
# 1) Squeeze-and-Excitation (3D)
# -----------------------------------------
class SELayer3D(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Hardsigmoid(inplace=True),
        )

    def forward(self, x):
        b, c, d, h, w = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y

# -----------------------------------------
# 2) Inverted Residual Block (3D)
# -----------------------------------------
class InvertedResidual3D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,               # (1,1,1) or (1,2,2)
        expand_ratio,
        use_se,
        activation
    ):
        super().__init__()
        hidden_dim = int(round(in_channels * expand_ratio))
        self.use_res_connect = (stride == (1,1,1) and in_channels == out_channels)

        if activation == "RE":
            act_layer = nn.ReLU(inplace=True)
        elif activation == "HS":
            act_layer = nn.Hardswish(inplace=True)
        else:
            raise NotImplementedError(f"Unknown activation {activation}")

        layers = []
        # 1) Pointwise expansion
        if expand_ratio != 1:
            layers.append(nn.Conv3d(in_channels, hidden_dim, 1, bias=False))
            layers.append(nn.BatchNorm3d(hidden_dim))
            layers.append(act_layer)
        # 2) Depthwise
        layers.append(
            nn.Conv3d(
                hidden_dim,
                hidden_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                groups=hidden_dim,
                bias=False
            )
        )
        layers.append(nn.BatchNorm3d(hidden_dim))
        if use_se:
            layers.append(SELayer3D(hidden_dim))
        layers.append(act_layer)
        # 3) Pointwise projection
        layers.append(nn.Conv3d(hidden_dim, out_channels, 1, bias=False))
        layers.append(nn.BatchNorm3d(out_channels))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

# -----------------------------------------
# 3) MobileNetV3-Large 3D Encoder
#    with 4 down-samples => final is 1/16
# -----------------------------------------
class MobileNetV3Large3DEncoder(nn.Module):
    """
    Stages:
      feats[0]:  input itself (no stride)
      feats[1]:  stem => [16] stride=(1,1,1)
      feats[2] & feats[3]: 1st down => [24]
      feats[4] & feats[5]: 2nd down => [40]
      feats[6] & feats[7]: 3rd down => [80]
      feats[8] & feats[9]: 4th down => [112]
      feats[10] & feats[11]: last blocks => [160]
      Total features => 12
    We'll skip from [1,3,5,7,9] for the 5-level UNet.
    The final is feats[11].
    """
    def __init__(self, in_channels=3):
        super().__init__()

        # We'll store input as feats[0]
        # 1) Stem => out=16, stride=(1,1,1)
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=3,
                      stride=(1,1,1), padding=1, bias=False),
            nn.BatchNorm3d(16),
            nn.Hardswish(inplace=True)
        )

        # Down 1 => out=24
        self.block2 = InvertedResidual3D(16, 24, 3, (1,2,2),
                                         expand_ratio=4, use_se=False, activation="RE")
        self.block3 = InvertedResidual3D(24, 24, 3, (1,1,1),
                                         expand_ratio=3, use_se=False, activation="RE")

        # Down 2 => out=40
        self.block4 = InvertedResidual3D(24, 40, 5, (1,2,2),
                                         expand_ratio=3, use_se=True, activation="HS")
        self.block5 = InvertedResidual3D(40, 40, 5, (1,1,1),
                                         expand_ratio=3, use_se=True, activation="HS")

        # Down 3 => out=80
        self.block6 = InvertedResidual3D(40, 80, 3, (1,2,2),
                                         expand_ratio=6, use_se=False, activation="HS")
        self.block7 = InvertedResidual3D(80, 80, 3, (1,1,1),
                                         expand_ratio=2.5, use_se=False, activation="HS")

        # Down 4 => out=112
        self.block8 = InvertedResidual3D(80, 112, 3, (1,2,2),
                                         expand_ratio=6, use_se=True, activation="HS")
        self.block9 = InvertedResidual3D(112, 112, 3, (1,1,1),
                                         expand_ratio=6, use_se=True, activation="HS")

        # Final blocks => out=160
        self.block10 = InvertedResidual3D(112, 160, 5, (1,1,1),
                                          expand_ratio=6, use_se=True, activation="HS")
        self.block11 = InvertedResidual3D(160, 160, 5, (1,1,1),
                                          expand_ratio=6, use_se=True, activation="HS")

    def forward(self, x):
        # feats[0] = the original input for skip connections
        feats = [x]

        # 1) Stem => feats[1]
        x = self.stem(x)
        feats.append(x)

        # Down 1 => feats[2] & feats[3]
        x = self.block2(x)
        feats.append(x)
        x = self.block3(x)
        feats.append(x)

        # Down 2 => feats[4] & feats[5]
        x = self.block4(x)
        feats.append(x)
        x = self.block5(x)
        feats.append(x)

        # Down 3 => feats[6] & feats[7]
        x = self.block6(x)
        feats.append(x)
        x = self.block7(x)
        feats.append(x)

        # Down 4 => feats[8] & feats[9]
        x = self.block8(x)
        feats.append(x)
        x = self.block9(x)
        feats.append(x)

        # Final => feats[10] & feats[11]
        x = self.block10(x)
        feats.append(x)
        x = self.block11(x)
        feats.append(x)

        return feats

# -----------------------------------------
# 4) DecoderBlock3D
# -----------------------------------------
class DecoderBlock3D(nn.Module):
    def __init__(self, in_c, out_c, skip_c):
        super().__init__()
        self.up = nn.ConvTranspose3d(
            in_channels=in_c,
            out_channels=out_c,
            kernel_size=(1,2,2),
            stride=(1,2,2)
        )
        self.conv = nn.Sequential(
            nn.Conv3d(out_c+skip_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)  # ups spatial by factor of 2
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='trilinear',
                              align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)

# -----------------------------------------
# 5) MobileNetV3-Large UNet 3D
#    Input [B, 3, T, 112, 112] => Output [B, 1, T, 112, 112]
# -----------------------------------------
class MobileNetV3UNet3D(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.encoder = MobileNetV3Large3DEncoder(in_channels)

        # We do 4 skip merges from:
        # feats[1] => [16], feats[3] => [24], feats[5] => [40], feats[7] => [80], feats[9] => [112].
        # Final encoder output is feats[11] => [160].
        # We'll decode in reversed order: feats[11] -> skip=feats[9] -> skip=feats[7] -> skip=feats[5] -> skip=feats[3] -> skip=feats[1].
        # feats[0] is the input itself if needed, but we typically only do 4 down-samplings => 4 ups.

        self.dec4 = DecoderBlock3D(in_c=160, out_c=112, skip_c=112)
        self.dec3 = DecoderBlock3D(in_c=112, out_c=80,  skip_c=80)
        self.dec2 = DecoderBlock3D(in_c=80,  out_c=40,  skip_c=40)
        self.dec1 = DecoderBlock3D(in_c=40,  out_c=24,  skip_c=24)

        # final up => merges with feats[1]
        self.final_up = nn.ConvTranspose3d(24, 16, (1,2,2), stride=(1,2,2))
        self.final_conv = nn.Sequential(
            nn.Conv3d(16+16, 16, 3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, out_channels, kernel_size=1)
        )

    def forward(self, x):
        feats = self.encoder(x)
        # feats[9] => [112], feats[7] => [80], feats[5] => [40], feats[3] => [24], feats[1] => [16].
        # feats[11] => [160]

        x = feats[11]                   # bottleneck => [160]
        x = self.dec4(x, feats[9])      # => [112]
        x = self.dec3(x, feats[7])      # => [80]
        x = self.dec2(x, feats[5])      # => [40]
        x = self.dec1(x, feats[3])      # => [24]

        # final up => compare with feats[1] => [16]
        x = self.final_up(x)
        if x.shape[2:] != feats[1].shape[2:]:
            x = F.interpolate(x, size=feats[1].shape[2:], mode='trilinear',
                              align_corners=False)
        # cat feats[1]
        x = torch.cat([feats[1], x], dim=1)
        x = self.final_conv(x)
        return x

# -----------------------------------------
# TEST
# -----------------------------------------
if __name__ == "__main__":
    net = MobileNetV3UNet3D(in_channels=3, out_channels=1).cuda()
    dummy = torch.randn(4, 3, 16, 112, 112).cuda()
    out = net(dummy)
    summary(model=net,input_size=(3, 16, 112, 112))
    print("Output shape:", out.shape)  # Expect [4, 1, 16, 112, 112]
