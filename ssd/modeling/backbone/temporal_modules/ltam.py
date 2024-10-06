import torch
from torch import nn
from ssd.modeling.rtdetr import AIFI, CrossAttention


class LTAM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        in_channels = out_channels = cfg.BACKBONE.OUT_CHANNELS[2]

        self.encoder = AIFI(reduce_dim=in_channels // 4, in_channel=in_channels, mlp_channel=out_channels)
        # Cross attention
        self.decoder = CrossAttention(dim=in_channels, frame_len=self.frame_len)

    def forward(self, enc_x, q_x):
        enc_x = self.encoder(enc_x)
        q_x = self.decoder(q_x)