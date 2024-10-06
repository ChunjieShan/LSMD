import torch
from torch import nn
from ssd.modeling.rtdetr import AIFI, CrossAttention


class STAM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.frame_len = cfg.DATASETS.FRAME_LENGTH
        # AIFI
        in_channels = out_channels = round(cfg.MODEL.BACKBONE.OUT_CHANNELS[2] * cfg.MODEL.BACKBONE.WIDTH)
        # in_channels = out_channels = 256
        self.in_features = ["dark3", "dark4", "dark5"]

        # self.encoder = AIFI(reduce_dim=in_channels // 4, is_3d=True, in_channel=in_channels)
        self.encoder = STAMEncoder(cfg, num_layers=1)

        # Cross attention
        # self.decoder = CrossAttention(dim=in_channels, frame_len=self.frame_len, is_attn=True)
        self.decoder = STAMDecoder(cfg, num_layers=1)

    def forward(self, x0=None, x1=None, x2=None):
        # if isinstance(xs, list) or isinstance(xs, tuple):
        #     x2, x1, x0 = xs
        # elif isinstance(xs, dict):
        #     features = [xs[f] for f in self.in_features]
        #     [x2, x1, x0] = features
        # elif isinstance(xs, torch.Tensor):
        #     x0, x1, x2 = xs, None, None
        # else:
        #     raise NotImplementedError("[E] Input tensor must be dictionary or list.")

        # if x0.ndim == 4:
        #     bt, c, h, w = x0.shape
        #     x0 = x0.view(-1, self.frame_len, c, h, w)
        #     x0 = x0.unsqueeze(0)  # make a fake bs dim to make onnx export happy

        x0 = self.encoder(x0)

        if x1 is not None:
            x0 = x0.unsqueeze(0)
            x2, x1, x0 = self.decoder(x2, x1, x0)
            if torch.jit.is_tracing():
                return [x2[:, -1, ...], x1[:, -1, ...], x0[:, -1, ...]]

            return [x2, x1, x0]

        else:
            n, c, h, w = x0.shape
            # x0 = x0.reshape(-1, self.frame_len, c, h, w)  # b, t, c, h, w
            # q_x, kv_x = x0[:, -1].permute(0, 2, 3, 1).flatten(1, 2), x0[:, :2].permute(0, 1, 3, 4, 2).flatten(1, 3)
            x2, x1, x0 = self.decoder(x2, x1, x0)

            return x0


class STAMDecoder(nn.Module):
    def __init__(self, cfg, num_layers=1):
        super().__init__()
        self.cfg = cfg
        self.frame_len = cfg.DATASETS.FRAME_LENGTH
        self.dim = int(cfg.MODEL.BACKBONE.OUT_CHANNELS[2] * cfg.MODEL.BACKBONE.WIDTH)
        self.blocks = nn.ModuleList()

        for i in range(num_layers):
            self.blocks.append(CrossAttention(frame_len=self.frame_len,
                                              dim=self.dim,
                                              is_attn=True))

    def forward(self, x2, x1, x0):
        n, c, h, w = x0.shape
        x0 = x0.reshape(-1, self.frame_len, c, h, w)  # b, t, c, h, w
        if x1 is not None:
            x1 = x1.reshape(-1, self.frame_len, x1.shape[1], x1.shape[2], x1.shape[3])
            x2 = x2.reshape(-1, self.frame_len, x2.shape[1], x2.shape[2], x2.shape[3])

        for block in self.blocks:
            x2, x1, x0 = block(x2, x1, x0)

        return x2[:, -1, ...] if x2 is not None else x2, x1[:, -1, ...] if x1 is not None else x1, x0[:, -1, ...]


class STAMEncoder(nn.Module):
    def __init__(self, cfg, num_layers=1):
        super().__init__()
        self.cfg = cfg
        self.frame_len = cfg.DATASETS.FRAME_LENGTH
        self.dim = int(cfg.MODEL.BACKBONE.OUT_CHANNELS[2] * cfg.MODEL.BACKBONE.WIDTH)
        self.blocks = nn.ModuleList()

        for i in range(num_layers):
            self.blocks.append(AIFI(self.dim // 4, in_channel=self.dim, is_3d=1024))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)

        return x


if __name__ == '__main__':
    import thop
    from ssd.config import cfg
    from ssd.data.build import build_transforms, build_target_transform

    cfg.merge_from_file(
        "../../../configs/resnet/dark/darknet_ssd320_dark_mem.yaml")
    cfg.freeze()
    dummy_x = torch.randn((1, 3, 1024, 10, 10))

    ltam = STAM(cfg)
    flops, params = thop.profile(ltam, inputs=(dummy_x))
    flops, params = thop.clever_format([flops, params], "%.3f")
    print(f"FLOPs: {flops}, Parameters: {params}")
