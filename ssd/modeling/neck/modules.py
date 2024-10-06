"""
modules.py - This file stores the rathering boring network blocks.
"""

import torch
import torch.nn as nn


class KeyProjection(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.key_proj = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        nn.init.orthogonal_(self.key_proj.weight.data)
        nn.init.zeros_(self.key_proj.bias.data)

    def forward(self, x):
        return self.key_proj(x)
