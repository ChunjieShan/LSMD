#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import torch
import torch.nn as nn

from typing import Dict
from ssd.modeling.backbone.common import C3, BaseConv, DWConv, Focus, SPPBottleneck, ResLayer, PAN
from ssd.modeling.backbone.attn_modules import ASPP
from ssd.modeling.stf import STF
# from ssd.modeling.swin import STAM
from ssd.modeling.temporal_modules import STAM, LTAM
from ssd.modeling import registry


class CSPLayer(nn.Module):
    """C3 in yolov5, CSP Bottleneck with 3 convolutions"""

    def __init__(
            self,
            in_channels,
            out_channels,
            n=1,
            shortcut=True,
            expansion=0.5,
            depthwise=False,
            act="silu",
    ):
        """
        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            n (int): number of Bottlenecks. Default value: 1.
        """
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        hidden_channels = int(out_channels * expansion)  # hidden channels
        self.conv1 = BaseConv(
            in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = BaseConv(
            in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv3 = BaseConv(2 * hidden_channels,
                              out_channels, 1, stride=1, act=act)
        module_list = [
            Bottleneck(
                hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act
            )
            for _ in range(n)
        ]
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = torch.cat((x_1, x_2), dim=1)
        return self.conv3(x)


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(
            self,
            in_channels,
            out_channels,
            shortcut=True,
            expansion=0.5,
            depthwise=False,
            act="silu",
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        Conv = DWConv if depthwise else BaseConv
        self.conv1 = BaseConv(
            in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = Conv(hidden_channels, out_channels, 3, stride=1, act=act)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y


class Darknet(nn.Module):
    # number of blocks from dark2 to dark5.
    depth2blocks = {21: [1, 2, 2, 1], 53: [2, 8, 8, 4]}

    def __init__(
        self,
        depth,
        in_channels=3,
        stem_out_channels=32,
        out_features=("dark3", "dark4", "dark5"),
    ):
        """
        Args:
            depth (int): depth of darknet used in model, usually use [21, 53] for this param.
            in_channels (int): number of input channels, for example, use 3 for RGB image.
            stem_out_channels (int): number of output channels of darknet stem.
                It decides channels of darknet layer2 to layer5.
            out_features (Tuple[str]): desired output layer name.
        """
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features
        self.stem = nn.Sequential(
            BaseConv(in_channels, stem_out_channels,
                     ksize=3, stride=1, act="lrelu"),
            *self.make_group_layer(stem_out_channels, num_blocks=1, stride=2),
        )
        in_channels = stem_out_channels * 2  # 64

        num_blocks = Darknet.depth2blocks[depth]
        # create darknet with `stem_out_channels` and `num_blocks` layers.
        # to make model structure more clear, we don't use `for` statement in python.
        self.dark2 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks[0], stride=2)
        )
        in_channels *= 2  # 128
        self.dark3 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks[1], stride=2)
        )
        in_channels *= 2  # 256
        self.dark4 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks[2], stride=2)
        )
        in_channels *= 2  # 512

        self.dark5 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks[3], stride=2),
            *self.make_spp_block([in_channels,
                                 in_channels * 2], in_channels * 2),
        )

    def make_group_layer(self, in_channels: int, num_blocks: int, stride: int = 1):
        "starts with conv layer then has `num_blocks` `ResLayer`"
        return [
            BaseConv(in_channels, in_channels * 2,
                     ksize=3, stride=stride, act="lrelu"),
            *[(ResLayer(in_channels * 2)) for _ in range(num_blocks)],
        ]

    def make_spp_block(self, filters_list, in_filters):
        m = nn.Sequential(
            *[
                BaseConv(
                    in_filters, filters_list[0], 1, stride=1, act="lrelu"),
                BaseConv(filters_list[0], filters_list[1],
                         3, stride=1, act="lrelu"),
                SPPBottleneck(
                    in_channels=filters_list[1],
                    out_channels=filters_list[0],
                    activation="lrelu",
                ),
                BaseConv(filters_list[0], filters_list[1],
                         3, stride=1, act="lrelu"),
                BaseConv(filters_list[1], filters_list[0],
                         1, stride=1, act="lrelu"),
            ]
        )
        return m

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        outputs["stem"] = x
        x = self.dark2(x)
        outputs["dark2"] = x
        x = self.dark3(x)
        outputs["dark3"] = x
        x = self.dark4(x)
        outputs["dark4"] = x
        x = self.dark5(x)
        outputs["dark5"] = x
        return {k: v for k, v in outputs.items() if k in self.out_features}


class CSPDarknet(nn.Module):
    def __init__(
        self,
        dep_mul,
        wid_mul,
        out_features=("dark3", "dark4", "dark5"),
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features
        self.width = wid_mul
        self.depth = dep_mul

        Conv = DWConv if depthwise else BaseConv

        base_channels = int(wid_mul * 64)  # 64
        base_depth = max(round(dep_mul * 3), 1)  # 3
        self.in_channels = [base_channels * 4,
                            base_channels * 8, base_channels * 16]

        # stem
        self.stem = Focus(3, base_channels, ksize=3, act=act)

        # dark2
        self.dark2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2, act=act),
            CSPLayer(
                base_channels * 2,
                base_channels * 2,
                n=base_depth,
                depthwise=depthwise,
                act=act,
            ),
        )

        # dark3
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2, act=act),
            CSPLayer(
                base_channels * 4,
                base_channels * 4,
                n=base_depth * 3,
                depthwise=depthwise,
                act=act,
            ),
        )

        # dark4
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2, act=act),
            CSPLayer(
                base_channels * 8,
                base_channels * 8,
                n=base_depth * 3,
                depthwise=depthwise,
                act=act,
            ),
        )

        # dark5
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, base_channels * 16, 3, 2, act=act),
            SPPBottleneck(base_channels * 16,
                          base_channels * 16, activation=act),
            CSPLayer(
                base_channels * 16,
                base_channels * 16,
                n=base_depth,
                shortcut=False,
                depthwise=depthwise,
                act=act,
            ),
        )

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        outputs["stem"] = x
        x = self.dark2(x)
        outputs["dark2"] = x
        x = self.dark3(x)
        outputs["dark3"] = x
        x = self.dark4(x)
        outputs["dark4"] = x
        x = self.dark5(x)
        outputs["dark5"] = x
        return {k: v for k, v in outputs.items() if k in self.out_features}


class CSPDarknetSTFPAN(nn.Module):
    def __init__(self, cfg, pretrained):
        super(CSPDarknetSTFPAN, self).__init__()
        # params
        self.in_features = ["dark3", "dark4", "dark5"]
        self.frame_len = cfg.DATASETS.FRAME_LENGTH

        # structures
        self.darknet = CSPDarknet(dep_mul=cfg.MODEL.BACKBONE.DEPTH,
                                  wid_mul=cfg.MODEL.BACKBONE.WIDTH,
                                  out_features=self.in_features)

        in_channels = self.darknet.in_channels
        self.in_channels = in_channels
        width, depth = self.darknet.width, self.darknet.depth

        # PAN
        self.pan = PAN(cfg)
        # ASPP
        # self.aspp = ASPP(in_channels=in_channels[0],
        #                  out_channels=in_channels[0],
        #                  atrous_rates=[1, 3, 5])

        # self.aspp_small = ASPP(in_channels=in_channels[2],
        #                        out_channels=in_channels[2],
        #                        atrous_rates=[1, 3, 5])

        # STF
        self.stam = STF(dim=1024, depth=1, heads=2, mlp_dim=2048, pe_dim=(10, 10), block_size=(5, 5),
                        frame_len=cfg.DATASETS.FRAME_LENGTH)

    def forward(self, x):
        b, t = x.shape[0:2]
        x = torch.flatten(x, start_dim=0, end_dim=1)
        outputs = self.darknet(x)
        features = [outputs[f] for f in self.in_features]
        [x2, x1, x0] = features

        # aggregates the middle head
        # x0 = x0.transpose(0, 1).unsqueeze(0)
        # x0 = self.stf(x0)
        # x1, x2 = x1[-1, ...].unsqueeze(0), x2[-1, ...].unsqueeze(0)

        # try to add ASPP module on the smallest feature map to make it attention to global tiny object
        # x0 = self.aspp_small(x0)

        # aggregates the smallest head
        c, h, w = x0.shape[1:]
        x0 = x0.view(b, t, c, h, w).transpose(1, 2)
        # x0 = x0.transpose(0, 1).unsqueeze(0)
        x0 = self.stam(x0)
        x1 = x1[(self.frame_len - 1)::self.frame_len]
        x2 = x2[(self.frame_len - 1)::self.frame_len]

        # x1, x2 = x1[-1, ...].unsqueeze(0), x2[-1, ...].unsqueeze(0)
        outputs = self.pan((x2, x1, x0))
        return outputs


# class CSPDarknetSTAMPAN(CSPDarknetSTFPAN):
#     def __init__(self, cfg, pretrained):
#         super().__init__(cfg, pretrained)
#         self.attn = STAM(in_channels=cfg.MODEL.BACKBONE.OUT_CHANNELS[2],
#                          num_heads=8,
#                          window_size=(3, 5, 5),
#                          depth=2)
#
#     def forward_features(self, x: torch.Tensor):
#         b, t = x.shape[0:2]
#         x = torch.flatten(x, start_dim=0, end_dim=1)
#         return self.darknet(x)
#
#     def forward_attn(self, x: torch.Tensor):
#         if x.ndim == 4:
#             c, h, w = x.shape[1:]
#             x = x.view(-1, self.frame_len, c, h, w).transpose(1, 2)
#         return self.attn(x)
#
#     def forward_pan(self, xs: Dict):
#         features = [xs[f] for f in self.in_features]
#         [x2, x1, x0] = features
#
#         x1 = x1[(self.frame_len - 1)::self.frame_len]
#         x2 = x2[(self.frame_len - 1)::self.frame_len]
#         # x1, x2 = x1[-1, ...].unsqueeze(0), x2[-1, ...].unsqueeze(0)
#         outputs = self.pan((x2, x1, x0))
#         return outputs
#
#     def forward_neck(self, xs: Dict):
#         features = [xs[f] for f in self.in_features]
#         b, t = features[0].shape[0:2]
#         [x2, x1, x0] = features
#
#         # aggregates the middle head
#         # x0 = x0.transpose(0, 1).unsqueeze(0)
#         # x0 = self.stf(x0)
#         # x1, x2 = x1[-1, ...].unsqueeze(0), x2[-1, ...].unsqueeze(0)
#
#         # try to add ASPP module on the smallest feature map to make it attention to global tiny object
#         # x0 = self.aspp_small(x0)
#
#         # aggregates the smallest head
#         c, h, w = x0.shape[1:]
#         x0 = x0.view(b, t, c, h, w).transpose(1, 2)
#         # x0 = x0.transpose(0, 1).unsqueeze(0)
#         x0, _ = self.attn(x0)
#         x1 = x1[(self.frame_len - 1)::self.frame_len]
#         x2 = x2[(self.frame_len - 1)::self.frame_len]
#         x0 = x0.squeeze(2)
#         # x1, x2 = x1[-1, ...].unsqueeze(0), x2[-1, ...].unsqueeze(0)
#         outputs = self.pan((x2, x1, x0))
#         return outputs
#
#     def forward(self, x):
#         xs = self.forward_features(x)
#         outputs = self.forward_neck(xs)
#
#         return outputs


class CSPDarknetSTAMPAN(CSPDarknetSTFPAN):
    def __init__(self, cfg, pretrained):
        super().__init__(cfg, pretrained)
        self.stam = STAM(cfg)

    def forward_features(self, x: torch.Tensor):
        b, t = x.shape[0:2]
        x = torch.flatten(x, start_dim=0, end_dim=1)
        return self.darknet(x)

    def forward_stam(self, xs):
        return self.stam(xs)

    def forward_pan(self, xs: Dict):
        features = [xs[f] for f in self.in_features]
        [x2, x1, x0] = features
        x1 = x1.reshape(-1, self.frame_len, x1.shape[1], x1.shape[2], x1.shape[3])
        x2 = x2.reshape(-1, self.frame_len, x2.shape[1], x2.shape[2], x2.shape[3])

        x1 = x1[:, -1, ...]
        x2 = x2[:, -1, ...]
        # x1, x2 = x1[-1, ...].unsqueeze(0), x2[-1, ...].unsqueeze(0)
        outputs = self.pan((x2, x1, x0))
        return outputs

    def forward_neck(self, xs: Dict):
        features = [xs[f] for f in self.in_features]
        b, t = features[0].shape[0:2]
        [x2, x1, x0] = features

        # aggregates the middle head
        # x0 = x0.transpose(0, 1).unsqueeze(0)
        # x0 = self.stf(x0)
        # x1, x2 = x1[-1, ...].unsqueeze(0), x2[-1, ...].unsqueeze(0)

        # try to add ASPP module on the smallest feature map to make it attention to global tiny object
        # x0 = self.aspp_small(x0)

        # aggregates the smallest head
        c, h, w = x0.shape[1:]
        x0 = x0.view(b, t, c, h, w).transpose(1, 2)
        # x0 = x0.transpose(0, 1).unsqueeze(0)
        x0, _ = self.stam(x0=x0)
        x1 = x1[(self.frame_len - 1)::self.frame_len]
        x2 = x2[(self.frame_len - 1)::self.frame_len]
        x0 = x0.squeeze(2)
        # x1, x2 = x1[-1, ...].unsqueeze(0), x2[-1, ...].unsqueeze(0)
        outputs = self.pan((x2, x1, x0))
        return outputs

    def forward(self, x):
        xs = self.forward_features(x)
        outputs = self.forward_neck(xs)

        return outputs

@registry.BACKBONES.register('darknet_stf_pan')
def darknet_stf_pan(cfg, pretrained=False):
    model = CSPDarknetSTFPAN(cfg, pretrained=pretrained)
    return model


@registry.BACKBONES.register('darknet_stam_pan')
def darknet_stam_pan(cfg, pretrained=False):
    model = CSPDarknetSTAMPAN(cfg, pretrained=pretrained)
    return model


if __name__ == '__main__':
    from ssd.config import cfg
    from ssd.data.build import build_transforms, build_target_transform

    cfg.merge_from_file(
        "../../../configs/resnet/dark/darknet_ssd320_dark_vid_fpn.yaml")
    cfg.freeze()
    darknet = CSPDarknetSTAMPAN(cfg, False)
    dummy_input = torch.randn((2, 3, 3, 320, 320))
    darknet(dummy_input)
