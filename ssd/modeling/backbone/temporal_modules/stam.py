import torch
from torch import nn
from ssd.modeling.rtdetr import AIFI, CrossAttention


class STAM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # AIFI
        in_channels = out_channels = cfg.BACKBONE.OUT_CHANNELS[2]
        self.in_features = ["dark3", "dark4", "dark5"]

        self.encoder = AIFI(reduce_dim=in_channels // 4, in_channel=in_channels, mlp_channel=out_channels)

        # Cross attention
        self.decoder = CrossAttention(dim=in_channels, frame_len=self.frame_len)

    def forward(self, xs):
        if isinstance(xs, list):
            x2, x1, x0 = xs
        elif isinstance(xs, dict):
            features = [xs[f] for f in self.in_features]
            [x2, x1, x0] = features
        else:
            raise NotImplementedError("[E] Input tensor must be dictionary or list.")

        x0 = self.encoder(x0)
        x2, x1, x0 = self.decoder(x2, x1, x0)

        return [x2, x1, x0]
