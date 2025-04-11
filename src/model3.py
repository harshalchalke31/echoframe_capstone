import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_large
from torchsummary import summary
from ptflops import get_model_complexity_info

class UpsampleBlock(nn.Module):
    """
    An upsampling block that:
      - Upsamples the input feature map to the spatial dimensions of a skip connection.
      - Concatenates the upsampled feature with the skip feature.
      - Applies two consecutive Conv-BatchNorm-ReLU operations.
    """
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # upsample x to match the spatial size of the skip connection
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        # concatenate along the channel dimension
        x = torch.cat([x, skip], dim=1)
        # two conv layers with BN and ReLU activation
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x

class MobileNetV3Encoder(nn.Module):
    """
    Pretrained MobileNetV3-large backbone used as the encoder.
    It returns features from several downsampling stages to serve as skip connections.
    """
    def __init__(self):
        super().__init__()
        self.backbone = mobilenet_v3_large(pretrained=True)
        # remove the classifier head
        self.backbone.classifier = nn.Identity()
        self.features = self.backbone.features
        
        # Define the indices corresponding to layers that downsample the input.
        self.downsample_indices = [0, 2, 3, 5, len(self.features) - 1]
        if 0 not in self.downsample_indices:
            self.downsample_indices.insert(0, 0)

    def forward(self, x: torch.Tensor):
        features = []
        # pass through each layer and store outputs at selected indices.
        for idx, layer in enumerate(self.features):
            x = layer(x)
            if idx in self.downsample_indices:
                features.append(x)
        return tuple(features)


class MobileNetV3UNet(nn.Module):
    """
    Segmentation model using a pretrained MobileNetV3-large encoder.
    The decoder fuses encoder skip connections with upsampling blocks.
    The network outputs a segmentation map with the desired number of classes.
    """
    def __init__(self, num_classes: int = 1):
        super().__init__()
        self.encoder = MobileNetV3Encoder()
        
        # retrieve the number of channels at each encoder stage via a dummy forward pass.
        self.feature_channels = self._get_encoder_channels()
        
        # build the decoder blocks.
        decoder_blocks = []
        in_channels = self.feature_channels[-1]  # The bottleneck channels
        # reverse iterate over encoder features (excluding the bottleneck) for skip connections.
        for skip_ch in reversed(self.feature_channels[:-1]):
            # define output channels as half of the skip connection channels (with a floor of 16)
            out_channels = max(16, skip_ch // 2)
            decoder_blocks.append(UpsampleBlock(in_channels, skip_ch, out_channels))
            in_channels = out_channels
        self.decoder_blocks = nn.ModuleList(decoder_blocks)
        
        # final upsampling and segmentation head (1x1 convolution).
        self.final_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.segmentation_head = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def _get_encoder_channels(self, input_size: tuple = (3, 224, 224)) -> list:
        """
        """
        with torch.no_grad():
            dummy_input = torch.randn(1, *input_size)
            feats = self.encoder(dummy_input)
        channels = [feat.size(1) for feat in feats]
        return channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder: obtain multi-scale features.
        encoder_features = self.encoder(x)
        x_dec = encoder_features[-1]  # start with the bottleneck features
        
        # Decoder: iteratively upsample and fuse skip connections.
        for idx, block in enumerate(self.decoder_blocks):
            # pick encoder features in reverse order (skipping the bottleneck).
            skip_feature = encoder_features[-(idx + 2)]
            x_dec = block(x_dec, skip_feature)
        
        # final upsampling and segmentation head.
        x_dec = self.final_upsample(x_dec)
        seg_map = self.segmentation_head(x_dec)
        return seg_map


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MobileNetV3UNet(num_classes=1).to(device)
    dummy_input = torch.randn(1, 3, 112, 112).to(device)
    output = model(dummy_input)
    
    print(model)
    summary(model, input_size=(3, 112, 112), device=str(device))
    print("Output shape:", output.shape)

    model.eval()

    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(model, (3, 112, 112), as_strings=True,
                                                print_per_layer_stat=False)

    print(f"MACs: {macs}")
    print(f"Params: {params}")
