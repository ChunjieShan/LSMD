"""
Creates a MobileNetV3 Model as defined in:
Andrew Howard, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang, Yukun Zhu, Ruoming Pang, Vijay Vasudevan, Quoc V. Le, Hartwig Adam. (2019).
Searching for MobileNetV3
arXiv preprint arXiv:1905.02244.


@ Credit from https://github.com/d-li14/mobilenetv3.pytorch
@ Modified by Chakkrit Termritthikun (https://github.com/chakkritte)

"""
import torch
import torch.nn as nn
import math

from ssd.modeling import registry
from ssd.utils.model_zoo import load_state_dict_from_url
from ssd.modeling.non_local import NONLocalBlock3D

model_urls = {
    'mobilenet_v3': 'https://github.com/d-li14/mobilenetv3.pytorch/raw/master/pretrained/mobilenetv3-large-1cd25616.pth',
}


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, _make_divisible(channel // reduction, 8)),
            nn.ReLU(inplace=True),
            nn.Linear(_make_divisible(channel // reduction, 8), channel),
            h_sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_bn, use_se, use_hs, use_nl, use_3d):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup

        if use_3d:
            use_se = False
            if inp == hidden_dim:
                self.conv = nn.Sequential(
                    # dw
                    nn.Conv3d(hidden_dim, hidden_dim, kernel_size, (stride, 1, 1), (kernel_size - 1) // 2, bias=False),
                    nn.BatchNorm3d(hidden_dim) if use_bn else nn.Identity(),
                    h_swish() if use_hs else nn.ReLU(inplace=True),
                    # Squeeze-and-Excite
                    SELayer(hidden_dim) if use_se else nn.Identity(),
                    # pw-linear
                    nn.Conv3d(hidden_dim, oup, 1, 1, 0, bias=False),
                    nn.BatchNorm3d(oup) if use_bn else nn.Identity(),
                    NONLocalBlock3D(oup, return_4d=False) if use_nl else nn.Identity(),
                )
            else:
                self.conv = nn.Sequential(
                    # pw
                    nn.Conv3d(inp, hidden_dim, 1, 1, 0, bias=False),
                    nn.BatchNorm3d(hidden_dim) if use_bn else nn.Identity(),
                    h_swish() if use_hs else nn.ReLU(inplace=True),
                    # dw
                    nn.Conv3d(hidden_dim, hidden_dim, kernel_size, (stride, 1, 1), (kernel_size - 1) // 2, bias=False),
                    nn.BatchNorm3d(hidden_dim) if use_bn else nn.Identity(),
                    # Squeeze-and-Excite
                    SELayer(hidden_dim) if use_se else nn.Identity(),
                    h_swish() if use_hs else nn.ReLU(inplace=True),
                    # pw-linear
                    nn.Conv3d(hidden_dim, oup, 1, 1, 0, bias=False),
                    nn.BatchNorm3d(oup) if use_bn else nn.Identity(),
                    NONLocalBlock3D(oup, return_4d=False) if use_nl else nn.Identity(),
                )

        else:
            if inp == hidden_dim:
                self.conv = nn.Sequential(
                    # dw
                    nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                    nn.BatchNorm2d(hidden_dim) if use_bn else nn.Identity(),
                    h_swish() if use_hs else nn.ReLU(inplace=True),
                    # Squeeze-and-Excite
                    SELayer(hidden_dim) if use_se else nn.Identity(),
                    # pw-linear
                    nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(oup) if use_bn else nn.Identity(),
                    NONLocalBlock3D(oup) if use_nl else nn.Identity(),
                )
            else:
                self.conv = nn.Sequential(
                    # pw
                    nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(hidden_dim) if use_bn else nn.Identity(),
                    h_swish() if use_hs else nn.ReLU(inplace=True),
                    # dw
                    nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                    nn.BatchNorm2d(hidden_dim) if use_bn else nn.Identity(),
                    # Squeeze-and-Excite
                    SELayer(hidden_dim) if use_se else nn.Identity(),
                    h_swish() if use_hs else nn.ReLU(inplace=True),
                    # pw-linear
                    nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(oup) if use_bn else nn.Identity(),
                    NONLocalBlock3D(oup) if use_nl else nn.Identity(),
                )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


# class MobileNetV3(nn.Module):
#     def __init__(self, mode='large', num_classes=1000, width_mult=1.):
#         super(MobileNetV3, self).__init__()
#         # setting of inverted residual blocks
#         self.cfgs = [
#             # k, t, c, SE, HS, NL, s
#             [3, 1, 16, 0, 0, 0, 1],
#             [3, 4, 24, 0, 0, 0, 2],
#             [3, 3, 24, 0, 0, 0, 1],
#             [5, 3, 40, 1, 0, 0, 2],
#             [5, 3, 40, 1, 0, 0, 1],
#             [5, 3, 40, 1, 0, 0, 1],
#             [3, 6, 80, 0, 1, 0, 2],
#             [3, 2.5, 80, 0, 1, 0, 1],
#             [3, 2.3, 80, 0, 1, 0, 1],
#             [3, 2.3, 80, 0, 1, 0, 1],
#             [3, 6, 112, 1, 1, 0, 1],
#             [3, 6, 112, 1, 1, 0, 1],
#             [5, 6, 160, 1, 1, 0, 2],
#             [5, 6, 160, 1, 1, 0, 1],
#             [5, 6, 160, 1, 1, 0, 1]]
#
#         assert mode in ['large', 'small']
#
#         # building first layer
#         input_channel = _make_divisible(16 * width_mult, 8)
#
#         layers = [conv_3x3_bn(3, input_channel, 2)]
#         # building inverted residual blocks
#         block = InvertedResidual
#         for k, t, c, use_se, use_hs, use_nl, s in self.cfgs:
#             output_channel = _make_divisible(c * width_mult, 8)
#             exp_size = _make_divisible(input_channel * t, 8)
#             use_se = False
#             layers.append(block(input_channel, exp_size, output_channel, k, s, True, use_se, use_hs, use_nl, False))
#             input_channel = output_channel
#         # building last several layers
#         layers.append(conv_1x1_bn(input_channel, exp_size))
#         self.features = nn.Sequential(*layers)
#         self.non_local = NONLocalBlock3D(in_channels=112)
#         self.avg_pool = nn.AvgPool3d((32, 1, 1))
#         self.extras = nn.ModuleList([
#             InvertedResidual(960, _make_divisible(960 * 0.2, 8), 512, 3, 2, True, True, True, False, False),
#             InvertedResidual(512, _make_divisible(512 * 0.25, 8), 256, 3, 2, True, True, True, False, False),
#             InvertedResidual(256, _make_divisible(256 * 0.5, 8), 256, 3, 2, True, True, True, False, False),
#             InvertedResidual(256, _make_divisible(256 * 0.25, 8), 64, 3, 2, False, True, True, False, False),
#         ])
#
#         self.reset_parameters()
#
#     def forward(self, x):
#         features = []
#
#         for i in range(13):
#             x = self.features[i](x)
#
#         # x = self.non_local(x)
#         x = self.convert_5d_tensor_to_4d(self.avg_pool(self.convert_4d_tensor_to_5d(x)))
#
#         features.append(x)
#         for i in range(13, len(self.features)):
#             x = self.features[i](x)
#         features.append(x)
#
#         for i in range(len(self.extras)):
#             x = self.extras[i](x)
#             features.append(x)
#
#         return tuple(features)
#
#     def reset_parameters(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#                 if m.bias is not None:
#                     m.bias.data.zero_()
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#             elif isinstance(m, nn.Linear):
#                 n = m.weight.size(1)
#                 m.weight.data.normal_(0, 0.01)
#                 m.bias.data.zero_()
#
#     @staticmethod
#     def convert_4d_tensor_to_5d(x: torch.Tensor):
#         """
#         B * C * H * W -> B * C * T * H * W
#         :param x:
#         :return:
#         """
#         assert x.dim() == 4, "[E] Input is not a 4D tensor!"
#         return x.transpose(0, 1).unsqueeze(0)
#
#     @staticmethod
#     def convert_5d_tensor_to_4d(x: torch.Tensor):
#         """
#         B * C * T * H * W -> B * C * H * W
#         :param x:
#         :return:
#         """
#         assert x.dim() == 5, "[E] Input is not a 5D tensor!"
#         return x.squeeze(0).transpose(0, 1)


class MobileNetV3(nn.Module):
    def __init__(self, mode='large', num_classes=1000, width_mult=1.):
        super(MobileNetV3, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = [
            # k, t, c, SE, HS, NL, s, 3d
            [3, 1, 16, 0, 0, 0, 1],
            [3, 4, 24, 0, 0, 0, 2],
            [3, 3, 24, 0, 0, 0, 1],
            [5, 3, 40, 1, 0, 0, 2],
            [5, 3, 40, 1, 0, 0, 1],
            [5, 3, 40, 1, 0, 0, 1],
            [3, 6, 80, 0, 1, 0, 2],
            [3, 2.5, 80, 0, 1, 0, 1],
            [3, 2.3, 80, 0, 1, 0, 1],
            [3, 2.3, 80, 0, 1, 0, 1],
            [3, 6, 112, 1, 1, 0, 1],
            [3, 6, 112, 1, 1, 0, 1],
            [5, 6, 160, 1, 1, 0, 2],
            [5, 6, 160, 1, 1, 0, 1],
            [5, 6, 160, 1, 1, 0, 1]]

        assert mode in ['large', 'small']

        # building first layer
        input_channel = _make_divisible(16 * width_mult, 8)

        layers = [conv_3x3_bn(3, input_channel, 2)]
        # building inverted residual blocks
        block = InvertedResidual
        for k, t, c, use_se, use_hs, use_nl, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            exp_size = _make_divisible(input_channel * t, 8)
            use_se = False
            layers.append(block(input_channel, exp_size, output_channel, k, s, True, use_se, use_hs, use_nl, False))
            input_channel = output_channel
        # building last several layers
        layers.append(conv_1x1_bn(input_channel, exp_size))
        self.features = nn.Sequential(*layers)
        self.non_local = NONLocalBlock3D(in_channels=112)
        self.avg_pool = nn.AvgPool3d((32, 1, 1))
        self.neck = nn.ModuleList([
            InvertedResidual(112, _make_divisible(112 * 0.25, 8), 112, 3, 2, True, False, True, True, True),
            InvertedResidual(112, _make_divisible(112 * 0.25, 8), 112, 3, 2, True, False, True, True, True),
            InvertedResidual(112, _make_divisible(112 * 0.25, 8), 112, 3, 2, True, False, True, True, True),
            InvertedResidual(112, _make_divisible(112 * 0.25, 8), 112, 3, 2, True, False, True, True, True),
            InvertedResidual(112, _make_divisible(112 * 0.25, 8), 112, 3, 2, True, False, True, True, True),
        ])
        self.extras = nn.ModuleList([
            InvertedResidual(960, _make_divisible(960 * 0.2, 8), 512, 3, 2, True, True, True, False, False),
            InvertedResidual(512, _make_divisible(512 * 0.25, 8), 256, 3, 2, True, True, True, False, False),
            InvertedResidual(256, _make_divisible(256 * 0.5, 8), 256, 3, 2, True, True, True, False, False),
            InvertedResidual(256, _make_divisible(256 * 0.25, 8), 64, 3, 2, False, True, True, False, False),
        ])

        self.reset_parameters()

    def forward(self, x):
        features = []

        for i in range(13):
            x = self.features[i](x)

        x = self.non_local(x)
        x = self.convert_4d_tensor_to_5d(x)
        # x = self.convert_5d_tensor_to_4d(self.avg_pool(x))

        for m in self.neck:
            x = m(x)

        x = self.convert_5d_tensor_to_4d(x)

        features.append(x)
        for i in range(13, len(self.features)):
            x = self.features[i](x)
        features.append(x)

        # for i in range(len(self.extras)):
        #     x = self.extras[i](x)
        #     features.append(x)

        return list(features)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    @staticmethod
    def convert_4d_tensor_to_5d(x: torch.Tensor):
        """
        B * C * H * W -> B * C * T * H * W
        :param x:
        :return:
        """
        assert x.dim() == 4, "[E] Input is not a 4D tensor!"
        return x.transpose(0, 1).unsqueeze(0)

    @staticmethod
    def convert_5d_tensor_to_4d(x: torch.Tensor):
        """
        B * C * T * H * W -> B * C * H * W
        :param x:
        :return:
        """
        assert x.dim() == 5, "[E] Input is not a 5D tensor!"
        return x.squeeze(0).transpose(0, 1)


@registry.BACKBONES.register('mobilenet_v3')
def mobilenet_v3(cfg, pretrained=False):
    model = MobileNetV3()
    if pretrained:
        model.load_state_dict(load_state_dict_from_url(model_urls['mobilenet_v3']), strict=False)
    return model