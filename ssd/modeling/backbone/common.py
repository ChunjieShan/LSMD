import torch
import torch.nn as nn


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = (
            d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
        )  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(
            c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = (
            self.default_act
            if act is True
            else act
            if isinstance(act, nn.Module)
            else nn.Identity()
        )

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(
        self, c1, c2, shortcut=True, g=1, e=0.5
    ):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.conv1 = Conv(c1, c_, 1, 1)
        self.conv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.conv2(self.conv1(x)) if self.add else self.conv2(self.conv1(x))


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(
        self, c1, c2, n=1, shortcut=True, g=1, e=0.5
    ):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.conv1 = Conv(c1, c_, 1, 1)
        self.conv2 = Conv(c1, c_, 1, 1)
        self.conv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(
            *(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n))
        )

    def forward(self, x):
        return self.conv3(torch.cat((self.m(self.conv1(x)), self.conv2(x)), 1))


class SiLU(nn.Module):
    """export-friendly version of nn.SiLU()"""

    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = nn.SiLU(inplace=inplace)
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


class BaseConv(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""

    def __init__(
        self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"
    ):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class DWConv(nn.Module):
    """Depthwise Conv + Conv"""

    def __init__(self, in_channels, out_channels, ksize, stride=1, act="silu"):
        super().__init__()
        self.dconv = BaseConv(
            in_channels,
            in_channels,
            ksize=ksize,
            stride=stride,
            groups=in_channels,
            act=act,
        )
        self.pconv = BaseConv(
            in_channels, out_channels, ksize=1, stride=1, groups=1, act=act
        )

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)


class ResLayer(nn.Module):
    "Residual layer with `in_channels` inputs."

    def __init__(self, in_channels: int):
        super().__init__()
        mid_channels = in_channels // 2
        self.layer1 = BaseConv(
            in_channels, mid_channels, ksize=1, stride=1, act="lrelu"
        )
        self.layer2 = BaseConv(
            mid_channels, in_channels, ksize=3, stride=1, act="lrelu"
        )

    def forward(self, x):
        out = self.layer2(self.layer1(x))
        return x + out


class SPPBottleneck(nn.Module):
    """Spatial pyramid pooling layer used in YOLOv3-SPP"""

    def __init__(
        self, in_channels, out_channels, kernel_sizes=(5, 9, 13), activation="silu"
    ):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=activation)
        self.m = nn.ModuleList(
            [
                nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
                for ks in kernel_sizes
            ]
        )
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = BaseConv(conv2_channels, out_channels, 1, stride=1, act=activation)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x


class Focus(nn.Module):
    """Focus width and height information into channel space."""

    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act="silu"):
        super().__init__()
        self.conv = BaseConv(in_channels * 4, out_channels, ksize, stride, act=act)

    def forward(self, x):
        # shape of x (b,c,w,h) -> y(b,4c,w/2,h/2)
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat(
            (patch_top_left, patch_bot_left, patch_top_right, patch_bot_right,), dim=1,
        )
        return self.conv(x)


class PAN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        width, depth = cfg.MODEL.BACKBONE.WIDTH, cfg.MODEL.BACKBONE.DEPTH
        in_channels = [256, 512, 1024]
        # self.box_head = build_box_head(cfg)

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(
            int(in_channels[2] * width), int(in_channels[1] * width), 1, 1
        )
        self.C3_p4 = C3(
            c1=int(2 * in_channels[1] * width),
            c2=int(in_channels[1] * width),
            n=round(3 * depth),
            shortcut=False,
        )  # cat

        self.reduce_conv1 = BaseConv(
            int(in_channels[1] * width), int(in_channels[0] * width), 1, 1
        )
        self.C3_p3 = C3(
            c1=int(2 * in_channels[0] * width),
            c2=int(in_channels[0] * width),
            n=round(3 * depth),
            shortcut=False,
        )

        # bottom-up conv
        self.bu_conv2 = BaseConv(
            int(in_channels[0] * width), int(in_channels[0] * width), 3, 2
        )
        self.C3_n3 = C3(
            c1=int(2 * in_channels[0] * width),
            c2=int(in_channels[1] * width),
            n=round(3 * depth),
            shortcut=False,
        )

        # bottom-up conv
        self.bu_conv1 = BaseConv(
            int(in_channels[1] * width), int(in_channels[1] * width), 3, 2
        )
        self.C3_n4 = C3(
            c1=int(2 * in_channels[1] * width),
            c2=int(in_channels[2] * width),
            n=round(3 * depth),
            shortcut=False,
        )

    # def forward(self, x0, x1, x2):
    def forward(self, xs):
        x2, x1, x0 = xs
        fpn_out0 = self.lateral_conv0(x0)  # 2048->1024/32
        f_out0 = self.upsample(fpn_out0)  # 1024/16
        f_out0 = torch.cat([f_out0, x1], 1)  # 1024->2048/16
        f_out0 = self.C3_p4(f_out0)  # 2048->1024/16

        fpn_out1 = self.reduce_conv1(f_out0)  # 1024->512/16
        f_out1 = self.upsample(fpn_out1)  # 512/8
        f_out1 = torch.cat([f_out1, x2], 1)  # 512->1024/8
        pan_out2 = self.C3_p3(f_out1)  # 1024->512/8

        # try to add ASPP module to make it attention to global tiny object
        # pan_out2 = self.aspp(pan_out2)

        p_out1 = self.bu_conv2(pan_out2)  # 512->512/16
        p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 512->1024/16
        pan_out1 = self.C3_n3(p_out1)  # 1024->1024/16

        p_out0 = self.bu_conv1(pan_out1)  # 1024->1024/32
        p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 1024->2048/32
        pan_out0 = self.C3_n4(p_out0)  # 2048->2048/32

        outputs = (pan_out2, pan_out1, pan_out0)
        # detections, _ = self.box_head(outputs)
        # boxes, scores, labels = detections
        #
        # return boxes, scores, labels

        return outputs
