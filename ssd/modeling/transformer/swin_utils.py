import torch
import torch.nn as nn
from ssd.modeling.non_local import NONLocalBlock3D


def window_partition(x, window_size: int):
    """
    Args:
        x: (B, C, T, H, W)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, C, T, H, W = x.shape
    x = x.view(B, C, T // window_size, window_size, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 1, 2, 4, 6, 3, 5, 7).contiguous().view(-1, C, window_size, window_size)
    return windows


def window_reverse(windows, window_size: int, B: int, T: int, H: int, W: int):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    N, C, _, _ = windows.shape
    x = windows.contiguous().view(B, C, T // window_size, H // window_size, W // window_size, window_size, window_size, window_size)
    x = x.permute(0, 1, 2, 5, 3, 6, 4, 7).contiguous().view(B, -1, T, H, W)
    return x


class WindowAttention(nn.Module):
    def __init__(self,
                 in_channels,
                 inner_channels=None,
                 out_channels=None):
        super(WindowAttention, self).__init__()
        self.in_channels = in_channels
        self.inner_channels = inner_channels if inner_channels else in_channels // 2
        self.out_channels = out_channels if out_channels else in_channels
        self.nl_block = NONLocalBlock3D(self.in_channels,
                                        self.inner_channels,
                                        self.out_channels)

    def forward(self, x, mask=None):
        return self.nl_block(x)


class SwinTransformerBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels=None,
                 window_size=None):
        super(SwinTransformerBlock, self).__init__()
        self.attn = WindowAttention(in_channels=in_channels,
                                    inner_channels=None,
                                    out_channels=out_channels)
        self.window_size = window_size

    def forward(self, x):
        B, C, T, H, W = x.shape
        x_windows = window_partition(x, window_size=self.window_size)
        attn_windows = self.attn(x_windows)
        shifted_x = window_reverse(attn_windows, self.window_size, B, T, H, W)

        return shifted_x


if __name__ == '__main__':
    dummy_input = torch.randn((1, 256, 16, 20, 20))
    model = SwinTransformerBlock(256, window_size=4)
    y = model(dummy_input)
    print(y.shape)
    # B, C, T, H, W = dummy_input.shape
    # window = window_partition(dummy_input, 4)
    # print(window.shape)
    # input = window_reverse(window, 4, 1, T, 20, 20)
    # print(input.shape)
