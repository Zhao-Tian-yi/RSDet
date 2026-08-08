"""
Custom convolution module for multi-scale feature fusion.
Replaces missing reference with a standard implementation.
"""

import torch
import torch.nn as nn


class Conv11Fusion4(nn.Module):
    """
    11-kernel convolution layer with ReLU activation.
    Designed to maintain spatial dimensions with padding=5.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, padding: int = 5):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=11, stride=stride, padding=padding)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x))
