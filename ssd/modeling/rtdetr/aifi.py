import torch
from torch import nn
from ssd.modeling.rtdetr.transformer import TransformerEncoderLayer
from ssd.modeling.stf.pos_embedding import positional_encoding_3d


class AIFI(TransformerEncoderLayer):
    def __init__(self, reduce_dim, in_channel, is_3d=True, mode="STAM", mlp_channel=2048, num_heads=8, dropout=0, act=nn.GELU(), normalize_before=False):
        super().__init__(in_channel, mlp_channel, num_heads, dropout, act, normalize_before)
        self.msa_in_channels = in_channel
        # self.msa_in_channels = reduce_dim
        # self.reduce_conv = nn.Conv2d(in_channel, reduce_dim, (1, 1))
        # self.up_conv = nn.Conv2d(reduce_dim, in_channel, (1, 1))
        self.is_3d = is_3d
        self.frame_len = 3
        self.mode = mode

    def forward(self, x):
        if x.ndim == 5:
            b, t, c, h, w = x.shape
            x = x.flatten(0, 1)
        else:
            bt, c, h, w = x.shape
            b, t = bt // self.frame_len, self.frame_len
        # x = self.reduce_conv(x)
        if not self.is_3d:
            pos_embed = self.build_2d_sincos_position_embedding(w, h, c)
            x = x.view(-1, h * w, c)
        else:
            if self.mode == "STAM":
                pos_embed = positional_encoding_3d(self.msa_in_channels, length=self.frame_len, height=h, width=w)
                pos_embed = pos_embed.permute(1, 2, 3, 0).flatten(0, 2)
            else:
                pos_embed = positional_encoding_3d(self.msa_in_channels, length=4, height=h, width=w)
                pos_embed = pos_embed.permute(1, 2, 3, 0).flatten(0, 2)

            x = x.view(-1, h * w * t, c)
        # flatten [B, C, H, W] to [B, HxW, C]
        # x = super().forward(x.flatten(2).permute(0, 2, 1), pos=pos_embed.to(device=x.device, dtype=x.dtype))
        x = super().forward(x, pos=pos_embed.to(device=x.device, dtype=x.dtype))
        x = x.permute(0, 2, 1).contiguous().view([-1, c, h, w])
        # x = self.up_conv(x)
        # return x.permute(0, 2, 1).view([-1, c, h, w]).contiguous()
        return x

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.):
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], 1)[None]




# if __name__ == '__main__':
#     # dummy_input = torch.randn(1, 1024, 10, 10)
#     # model = AIFI(reduce_dim=256, in_channel=1024, mlp_channel=1024)
#     # model(dummy_input)
#     from ultralytics import RTDETR
#     model = RTDETR("/mnt/h/Code/3.Projects/3.Video-Understanding/2.VID/23-06-30-SSD-STAM/checkpoints/rtdetr-l.pt")
#     print(model)
#     model("/mnt/h/Dataset/2.Carotid-Artery/2.Object-Detection/1.Training/20230825/images/train/4BB4C786-17E2-4C93-9449-999F3DB481A9/img_00001.jpg")


if __name__ == '__main__':
    from ssd.modeling.rtdetr import CrossAttention
    dummy_q = torch.randn((1, 100, 1024))
    dummy_kv = torch.randn((1, 2, 100, 1024))
    model = CrossAttention(dim=1024)
    model(dummy_q, dummy_kv)
