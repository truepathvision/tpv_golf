"""Minimal FRAN U-Net architecture for face re-aging inference.

Only imported when FRAN weights are available and torch is installed.
Architecture follows the timroelofs123/face_reaging implementation.
"""

import torch
import torch.nn as nn


class _DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class FRANUNet(nn.Module):
    """U-Net that takes an image + source/target age scalars and outputs a re-aged image."""

    def __init__(self, in_channels: int = 5, out_channels: int = 3, features: list[int] | None = None):
        super().__init__()
        if features is None:
            features = [64, 128, 256, 512]

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)

        prev = in_channels
        for f in features:
            self.downs.append(_DoubleConv(prev, f))
            prev = f

        self.bottleneck = _DoubleConv(features[-1], features[-1] * 2)

        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(f * 2, f, 2, stride=2))
            self.ups.append(_DoubleConv(f * 2, f))

        self.final = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x: torch.Tensor, age_in: torch.Tensor, age_out: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        age_in_map = age_in.view(b, 1, 1, 1).expand(b, 1, h, w)
        age_out_map = age_out.view(b, 1, 1, 1).expand(b, 1, h, w)
        x = torch.cat([x, age_in_map, age_out_map], dim=1)

        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skips = skips[::-1]

        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip = skips[i // 2]
            if x.shape != skip.shape:
                x = torch.nn.functional.interpolate(x, size=skip.shape[2:])
            x = torch.cat([skip, x], dim=1)
            x = self.ups[i + 1](x)

        return torch.sigmoid(self.final(x))
