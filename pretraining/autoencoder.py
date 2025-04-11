import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model11 import MobileNetV3Large3DEncoder,get_norm_3d


class AutoDecoder3D(nn.Module):
    def __init__(self, in_channels=160, out_channels=3, dropout=0.0, norm_type="bn"):
        super().__init__()

        def block(in_c, out_c):
            return nn.Sequential(
                nn.ConvTranspose3d(in_c, out_c, kernel_size=(1, 2, 2), stride=(1, 2, 2)),
                get_norm_3d(out_c, norm_type),
                nn.ReLU(inplace=True),
                nn.Dropout3d(p=dropout) if dropout > 0.0 else nn.Identity(),
                nn.Conv3d(out_c, out_c, kernel_size=3, padding=1),
                get_norm_3d(out_c, norm_type),
                nn.ReLU(inplace=True),
            )

        self.decoder = nn.Sequential(
            block(160, 112),  # 7x7 → 14x14
            block(112, 80),   # 14x14 → 28x28
            block(80, 40),    # 28x28 → 56x56
            block(40, 24),    # 56x56 → 112x112
            nn.Conv3d(24, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(x)


class MobileNetV3AutoEncoder3D(nn.Module):
    def __init__(self, in_channels=3, dropout=0.0, norm_type="bn"):
        super().__init__()
        self.encoder = MobileNetV3Large3DEncoder(in_channels, norm_type=norm_type)
        self.decoder = AutoDecoder3D(in_channels=160, out_channels=in_channels,
                                     dropout=dropout, norm_type=norm_type)

    def forward(self, x):
        feats = self.encoder(x)
        x = feats[self.encoder.final_idx]
        return self.decoder(x)

if __name__=='__main__':
    model = MobileNetV3AutoEncoder3D(in_channels=3, dropout=0.1).cuda()
    x = torch.randn(2, 3, 16, 112, 112).cuda()
    out = model(x)
    print("Input shape:", x.shape)
    print("Output shape:", out.shape)

