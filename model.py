"""
model.py - Custom Lane Segmentation Neural Network Architecture
Designed and implemented from scratch for PSU-reservoir lane dataset.

Architecture Overview:
    - Lightweight Encoder-Decoder with Skip Connections (Custom U-Net style)
    - Designed specifically for low memory footprint on local machines (laptop/PC)
    - Total parameters: ~400K parameters (< 2 MB weights)
    - Supports flexible input resolutions (e.g., 64x36, 128x72, 128x128)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    Custom Double Convolution Block:
    [Conv2d -> BatchNorm2d -> LeakyReLU] x 2
    """
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super(ConvBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecoderBlock(nn.Module):
    """
    Custom Decoder Block with Skip Connection:
    Upsample (Bilinear/Nearest or ConvTranspose2d) -> Concat(skip) -> ConvBlock
    Uses bilinear interpolation to guarantee exact spatial dimension matching
    regardless of odd/even dimension division.
    """
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super(DecoderBlock, self).__init__()
        self.reduce_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv_block = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # Upsample to match skip connection spatial dimensions exactly
        x = F.interpolate(x, size=(skip.shape[2], skip.shape[3]), mode='bilinear', align_corners=False)
        x = self.reduce_conv(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv_block(x)


class CustomLaneSegNet(nn.Module):
    """
    CustomLaneSegNet: Custom-designed Lightweight CNN for Lane Segmentation.
    
    Layers:
        Encoder 1: 3  -> 16 channels, downsample /2
        Encoder 2: 16 -> 32 channels, downsample /2
        Encoder 3: 32 -> 64 channels, downsample /2
        Bottleneck: 64 -> 128 channels with dropout
        Decoder 3: (128 -> 64) + skip(64) -> 64 channels
        Decoder 2: (64  -> 32) + skip(32) -> 32 channels
        Decoder 1: (32  -> 16) + skip(16) -> 16 channels
        Output Head: 16 -> 1 channel (logits)
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 1):
        super(CustomLaneSegNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # --- Encoder ---
        self.enc1 = ConvBlock(in_channels, 16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc2 = ConvBlock(16, 32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc3 = ConvBlock(32, 64)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- Bottleneck ---
        self.bottleneck = ConvBlock(64, 128, dropout=0.2)

        # --- Decoder ---
        self.dec3 = DecoderBlock(in_channels=128, skip_channels=64, out_channels=64)
        self.dec2 = DecoderBlock(in_channels=64, skip_channels=32, out_channels=32)
        self.dec1 = DecoderBlock(in_channels=32, skip_channels=16, out_channels=16)

        # --- Output Head ---
        self.out_conv = nn.Conv2d(16, num_classes, kernel_size=1)

        # Initialize weights from scratch using Kaiming Normal
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder forward pass with skip connections
        s1 = self.enc1(x)
        p1 = self.pool1(s1)

        s2 = self.enc2(p1)
        p2 = self.pool2(s2)

        s3 = self.enc3(p2)
        p3 = self.pool3(s3)

        # Bottleneck
        b = self.bottleneck(p3)

        # Decoder forward pass with concatenated skips
        d3 = self.dec3(b, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)

        # Output logits
        logits = self.out_conv(d1)
        return logits


def get_model(in_channels: int = 3, num_classes: int = 1) -> CustomLaneSegNet:
    """Helper to instantiate and return the custom model."""
    return CustomLaneSegNet(in_channels=in_channels, num_classes=num_classes)


if __name__ == '__main__':
    # Unit test & architecture summary
    model = get_model()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("=" * 60)
    print("CustomLaneSegNet Architecture Verification")
    print("=" * 60)
    print(f"Total Parameters:     {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Model Size (FP32):    {total_params * 4 / (1024 * 1024):.2f} MB")

    # Test with standard 16:9 input sizes
    test_sizes = [(36, 64), (72, 128), (144, 256), (32, 32), (48, 48)]
    for h, w in test_sizes:
        dummy_input = torch.randn(2, 3, h, w)
        output = model(dummy_input)
        print(f"Input: {list(dummy_input.shape)} -> Output: {list(output.shape)} [OK]")
    print("=" * 60)
