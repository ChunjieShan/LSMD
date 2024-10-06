import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.checkpoint as checkpoint

from ssd.modeling.stf.pos_embedding import positional_encoding_3d
from ssd.modeling.stf.transformer_block import (
    TransformerEncoder,
    TransformerEncoderLayer,
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerHELayer
)


class STF(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, pe_dim, block_size=None, frame_len=None):
        super().__init__()
        if block_size is None:
            block_size = [16, 16]

        if frame_len is None:
            frame_len = 3

        self.frame_len = frame_len
        dec_frame_len = 3
        avg_kernel_size = frame_len // dec_frame_len
        enc_layer = TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, block_size=block_size)
        self.encoder = TransformerEncoder(enc_layer, num_layers=depth)
        dec_layer = TransformerDecoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, block_size=block_size)
        self.decoder = TransformerDecoder(dec_layer, num_layers=depth)

        self.pe_enc = nn.Parameter(torch.rand([1, dim, frame_len, pe_dim[0], pe_dim[1]]), requires_grad=True)
        self.pe_dec = nn.Parameter(torch.rand([1, dim, 1, pe_dim[0], pe_dim[1]]), requires_grad=True)
        # self.pe_kv_dec = nn.Parameter(torch.rand([1, dim, dec_frame_len, pe_dim[0], pe_dim[1]]), requires_grad=True)
        self.avg_pool = nn.AvgPool3d(kernel_size=(avg_kernel_size, 1, 1))

    def forward(self, x_enc):
        x_enc_last = x_enc[:, :, -1, ...].unsqueeze(2)
        n, c, t_e, h, w = x_enc.shape

        pe_enc = self.pe_enc
        pe_dec = self.pe_dec

        if pe_enc.shape[3] != h or pe_enc.shape[4] != w:
            pe_enc = F.interpolate(pe_enc, size=(self.frame_len, h, w), mode='trilinear')
            pe_dec = F.interpolate(pe_dec, size=(1, h, w), mode='trilinear')

        if n > 1:
            pe_enc = pe_enc.repeat(n, 1, 1, 1, 1)
            pe_dec = pe_dec.repeat(n, 1, 1, 1, 1)

        x_enc = self.encoder(x_enc, pos=pe_enc)
        # x_enc = self.avg_pool(x_enc)
        x_dec = self.decoder(x_enc_last, x_enc, pos=pe_enc, query_pos=pe_dec)
        x_dec = x_dec.squeeze(2)

        return x_dec


class MAR(nn.Module):
    def __init__(self, dim, mlp_dim, KL, KH, num_class):
        super().__init__()
        self.HEdecoder = TransformerHELayer(d_model=dim, dim_feedforward=dim // 2)
        self.memory_key = nn.Parameter(torch.rand([1, KL * num_class, dim]), requires_grad=True)
        self.memory_value = nn.Parameter(torch.rand([1, KH * num_class, dim]), requires_grad=True)

    def forward(self, x):
        n, c, h, w = x.shape

        memory_key = self.memory_key.repeat([n, 1, 1])
        memory_value = self.memory_value.repeat([n, 1, 1])

        x = self.HEdecoder(query=x, key=memory_key, value=memory_value)

        return x


class MultiScaleSTF(nn.Module):
    def __init__(self,
                 in_channels,
                 frame_len,
                 expansion=2):
        super(MultiScaleSTF, self).__init__()
        self.in_channels = in_channels
        self.stf1 = STF(dim=in_channels[0], depth=1, heads=2, mlp_dim=in_channels[0] * expansion, pe_dim=(40, 40),
                        block_size=(8, 8),
                        frame_len=frame_len)
        self.stf2 = STF(dim=in_channels[1], depth=1, heads=2, mlp_dim=in_channels[1] * expansion, pe_dim=(20, 20),
                        block_size=(4, 4),
                        frame_len=frame_len)
        self.stf3 = STF(dim=in_channels[2], depth=1, heads=2, mlp_dim=in_channels[2] * expansion, pe_dim=(10, 10),
                        block_size=(2, 2),
                        frame_len=frame_len)
        self.m = nn.ModuleList([self.stf1, self.stf2, self.stf3])

    def forward(self, xs):
        if not isinstance(xs, list) and not isinstance(xs, tuple):
            xs = [xs]

        outputs = []
        for i, feat in enumerate(xs):
            if feat.ndim == 4:
                feat = feat.transpose(0, 1).unsqueeze(0)
            out = self.m[i](feat)
            outputs.append(out)

        return outputs


if __name__ == '__main__':
    import random
    import numpy as np

    model = STF(dim=2048, depth=1, heads=2, mlp_dim=256, pe_dim=(10, 10), block_size=(2, 2), frame_len=3)
    model.cuda().eval()

    x = torch.rand(2, 2048, 3, 10, 10).cuda()
    torch.save(model.state_dict(), "./stf.pth")
    #     with torch.no_grad():
    # while True:
    #     preds = model(x)
    #     print(preds.shape)

#     model = MAR(dim=512, mlp_dim=256, KL=10, KH=10, num_class=19)
#     model.cuda().eval()

#     x = torch.rand(1, 512, 128, 256).cuda()
#     with torch.no_grad():
#         preds = model(x)
#         print(preds.shape)
