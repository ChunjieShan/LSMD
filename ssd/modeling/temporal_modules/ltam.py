import torch
import math
from torch import nn
from ssd.modeling.rtdetr import AIFI, CrossAttention


class LTAM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.is_attn = True
        width = cfg.MODEL.BACKBONE.WIDTH
        in_channels = round(cfg.MODEL.BACKBONE.OUT_CHANNELS[2] * width)
        # in_channels = 256
        # out_channels = round(cfg.MODEL.BACKBONE.OUT_CHANNELS[2] * width)
        self.cfg = cfg
        self.frame_len = cfg.DATASETS.FRAME_LENGTH

        self.encoder = AIFI(reduce_dim=256, is_3d=True, mode="LTAM", in_channel=in_channels, mlp_channel=2048)
        self.conv3d_trans = nn.Conv3d(in_channels=round(in_channels * width),
                                      out_channels=round(in_channels * width),
                                      kernel_size=(4, 3, 3),
                                      padding=(0, 1, 1))
        # Cross attention
        # self.decoder = CrossAttention(dim=in_channels, frame_len=self.frame_len, is_attn=False)
        self.decoder = CrossAttention(dim=in_channels, frame_len=self.frame_len, is_attn=self.is_attn)

    def forward(self, enc_x, q_x):
        # b, l, c = q_x.shape
        # h = w = int(math.sqrt(l))
        # enc_x = enc_x.view(-1, c, h, w)
        # enc_x = self.encoder(enc_x)
        # enc_x = enc_x.view(b, -1, c)
        b, t, c, h, w = enc_x.shape
        if not self.is_attn:
            enc_feat = self.encoder(enc_x).contiguous().view(b, t, c, h, w)
            x0 = torch.cat([enc_feat, q_x], dim=1)
            _, _, q_x = self.decoder(x0=x0)
        else:
            enc_feat = torch.cat([enc_x, q_x], dim=1)
            t_enc = enc_feat.shape[1]
            enc_feat = self.encoder(enc_feat).contiguous().view(b, c, t_enc, h, w)
            enc_feat = self.conv3d_trans(enc_feat).transpose(1, 2)
            x0 = torch.cat([enc_feat, q_x], dim=1)
            _, _, q_x = self.decoder(x0=x0)

        if torch.jit.is_tracing():
            if q_x.ndim == 5:
                q_x = q_x.squeeze(1)
            else:
                q_x = q_x.permute(0, 2, 1).reshape(-1, int(
                    self.cfg.MODEL.BACKBONE.OUT_CHANNELS[2] * self.cfg.MODEL.BACKBONE.WIDTH), 10, 10)

            if enc_feat.ndim == 5:
                enc_feat = enc_feat.squeeze(1)

        return enc_feat, q_x


if __name__ == '__main__':
    import thop
    from ssd.config import cfg
    from ssd.data.build import build_transforms, build_target_transform

    cfg.merge_from_file(
        "../../../configs/resnet/dark/darknet_ssd320_dark_mem.yaml")
    cfg.freeze()
    dummy_enc = torch.randn((1, 3, 256, 40, 40)).to("cuda:0")
    dummy_x = torch.randn((1, 1, 256, 40, 40)).to("cuda:0")

    ltam = LTAM(cfg).to("cuda:0")
    flops, params = thop.profile(ltam, inputs=(dummy_enc, dummy_x))
    flops, params = thop.clever_format([flops, params], "%.3f")
    print(f"FLOPs: {flops}, Parameters: {params}")
