import torch
import torch.nn as nn
from typing import List, Dict
from torchvision.models import resnet50, ResNet50_Weights
from ssd.modeling import registry
from ssd.modeling.swin import STAM
from ssd.modeling.stf import STF
from ssd.modeling.backbone.common import C3, Conv
from torchvision.models._utils import IntermediateLayerGetter


class ResNet50(nn.Module):
    def __init__(self, cfg, pretrained):
        super(ResNet50, self).__init__()
        if pretrained:
            self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            print("Loading ResNet-50 pretrained weights...")
        else:
            self.resnet = resnet50(weights=None)
            print("No pretrained ResNet-50 weights.")

        self.conv1 = self.resnet.conv1
        self.bn1 = self.resnet.bn1
        self.relu = self.resnet.relu  # 1/2, 64
        self.maxpool = self.resnet.maxpool

        self.layer1 = self.resnet.layer1  # 1/4, 256
        self.layer2 = self.resnet.layer2  # 1/8, 512
        self.layer3 = self.resnet.layer3  # 1/16, 1024
        self.layer4 = self.resnet.layer4  # 1/32, 2048

        # self.non_local = NONLocalBlock3D(in_channels=1024, need_det_token=False)
        # self.nl1 = NONLocalBlock3D(in_channels=1024, need_det_token=False)
        # self.nl2 = NONLocalBlock3D(in_channels=512, need_det_token=False)
        # self.nl3 = NONLocalBlock3D(in_channels=256, need_det_token=False)
        # self.swin1 = SwinTrans3D(dim=2048,
        #                          depth=2,
        #                          num_heads=8,
        #                          window_size=(8, 4, 4))
        # self.swin2 = SwinTrans3D(dim=512,
        #                          depth=2,
        #                          num_heads=8,
        #                          window_size=(8, 2, 2))
        # self.swin3 = SwinTrans3D(dim=256,
        #                          depth=2,
        #                          num_heads=8,
        #                          window_size=(8, 1, 1))

        num_frames = cfg.DATASETS.FRAME_LENGTH
        self.reduce_conv = C3(c1=2048, c2=1024)
        # self.stf = STF(dim=1024, depth=1, heads=2, mlp_dim=256, pe_dim=(10, 10), block_size=(2, 2), frame_len=num_frames)
        self.avg_pool = nn.AvgPool3d((num_frames, 1, 1))
        # self.conv2 = nn.Conv2d(2048, 512, (3, 3), (2, 2), (1, 1))  # 1/32
        # self.conv3 = nn.Conv2d(512, 256, (3, 3), (2, 2), (1, 1))  # 1/64
        # self.conv4 = nn.Conv2d(256, 256, (3, 3), (2, 2), (1, 1))  # 1/128
        # self.conv5 = nn.Conv2d(256, 256, (3, 3), (2, 2), (1, 1))  # 1/256
        # self.conv6 = nn.Conv2d(256, 64, (3, 3), (2, 2), (1, 1))  # 1/256

    def forward_features(self, x):
        x = self.maxpool(self.bn1(self.conv1(x)))
        x = self.relu(x)  # 1/2, 64

        x = self.layer1(x)  # 1/4, 256
        x = self.layer2(x)  # 1/8, 512
        x = self.layer3(x)  # 1/16, 1024
        x = self.layer4(x)  # 1/32, 2048

        return x

    def forward(self, x):
        features = []
        x = self.forward_features(x)
        x = self.reduce_conv(x)
        # x = x[-1, :].unsqueeze(0)
        x = self.convert_4d_tensor_to_5d(x)
        # x = self.avg_pool(x)
        # x = self.convert_5d_tensor_to_4d(x)

        # x1, x2, x3 = x.unbind(0)
        # x1 = x1.unsqueeze(1).unsqueeze(0)
        # x2 = x2.unsqueeze(1).unsqueeze(0)
        # x3 = x3.unsqueeze(1).unsqueeze(0)
        # feat = self.stf(x)

        # feat = self.nl1(x)[-1, :].unsqueeze(0)
        # feat = self.convert_4d_tensor_to_5d(x)
        # feat = self.swin1(feat)
        # feat = self.convert_5d_tensor_to_4d(feat)
        # feat = feat[-1, :].unsqueeze(0)
        # features.append(feat)

        # feat = x[-1, :].unsqueeze(0)
        features.append(x)

        # x = self.conv2(x)
        # feat = self.nl2(x)[-1, :].unsqueeze(0)
        # feat = self.convert_4d_tensor_to_5d(x)
        # feat = self.swin2(feat)
        # feat = self.convert_5d_tensor_to_4d(feat)
        # feat = feat[-1, :].unsqueeze(0)
        # features.append(feat)

        # feat = x[-1, :].unsqueeze(0)
        # features.append(feat)
        # features.append(x)

        # x = self.conv3(x)
        # feat = self.nl3(x)[-1, :].unsqueeze(0)
        # feat = self.convert_4d_tensor_to_5d(x)
        # feat = self.swin3(feat)
        # feat = self.convert_5d_tensor_to_4d(feat)
        # feat = feat[-1, :].unsqueeze(0)
        # features.append(feat)

        # feat = x[-1, :].unsqueeze(0)
        # features.append(feat)

        # features.append(x)
        # x = self.conv4(x)
        # feat = x[-1, :].unsqueeze(0)
        # features.append(feat)
        # features.append(x)
        # x = self.conv5(x)
        # feat = x[-1, :].unsqueeze(0)
        # features.append(feat)
        # x = self.conv6(x)
        # feat = x[-1, :].unsqueeze(0)
        # features.append(feat)
        # features.append(x)

        return features

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


class ResNetMultiOutputs(ResNet50):
    def __init__(self, cfg, pretrained):
        super(ResNetMultiOutputs, self).__init__(cfg, pretrained)
        self.in_channels = [512, 1024, 2048]
        self.width = 1.0
        self.depth = 1.0

    def forward_features(self, x):
        outputs = {}
        if x.ndim == 5:
            b, t, c, h, w = x.shape
            x = x.view(b * t, c, h, w)
        x = self.maxpool(self.bn1(self.conv1(x)))
        x = self.relu(x)  # 1/2, 64

        x = self.layer1(x)  # 1/4, 256
        x = self.layer2(x)  # 1/8, 512
        outputs["layer2"] = x
        x = self.layer3(x)  # 1/16, 1024
        outputs["layer3"] = x
        x = self.layer4(x)  # 1/32, 2048
        outputs["layer4"] = x

        return outputs

    def forward(self, x):
        outputs = self.forward_features(x)
        return outputs


class ResNetPAN(nn.Module):
    def __init__(self, cfg, pretrained):
        super(ResNetPAN, self).__init__()
        self.resnet = ResNetMultiOutputs(cfg, pretrained)

        # params
        self.in_features = ["layer2", "layer3", "layer4"]
        in_channels = self.resnet.in_channels
        self.in_channels = in_channels
        width, depth = self.resnet.width, self.resnet.depth
        self.frame_len = cfg.DATASETS.FRAME_LENGTH
        self.bs = cfg.SOLVER.BATCH_SIZE

        # structures
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = Conv(
            int(in_channels[2] * width), int(in_channels[1] * width), 1, 1
        )
        self.C3_p4 = C3(
            c1=int(2 * in_channels[1] * width),
            c2=int(in_channels[1] * width),
            n=round(3 * depth),
            shortcut=False,
        )  # cat

        self.reduce_conv1 = Conv(
            int(in_channels[1] * width), int(in_channels[0] * width), 1, 1
        )
        self.C3_p3 = C3(
            c1=int(2 * in_channels[0] * width),
            c2=int(in_channels[0] * width),
            n=round(3 * depth),
            shortcut=False,
        )

        # bottom-up conv
        self.bu_conv2 = Conv(
            int(in_channels[0] * width), int(in_channels[0] * width), 3, 2
        )
        self.C3_n3 = C3(
            c1=int(2 * in_channels[0] * width),
            c2=int(in_channels[1] * width),
            n=round(3 * depth),
            shortcut=False,
        )

        # bottom-up conv
        self.bu_conv1 = Conv(
            int(in_channels[1] * width), int(in_channels[1] * width), 3, 2
        )
        self.C3_n4 = C3(
            c1=int(2 * in_channels[1] * width),
            c2=int(in_channels[2] * width),
            n=round(3 * depth),
            shortcut=False,
        )
        self.stf = STF(dim=2048, depth=1, heads=2, mlp_dim=2048, pe_dim=(10, 10), block_size=(5, 5),
                       frame_len=cfg.DATASETS.FRAME_LENGTH)

    def forward_features(self, x: torch.Tensor):
        """

        :param x: torch.Tensor
        :return: dict
        """
        return self.resnet(x)

    def forward_stf(self, x: torch.Tensor):
        c, h, w = x.shape[1:]
        if x.ndim == 4:
            x = x.view(-1, self.frame_len, c, h, w).transpose(1, 2)
        return self.stf(x)

    def forward_pan(self, xs: Dict):
        features = [xs[f] for f in self.in_features]
        [x2, x1, x0] = features

        # c, h, w = x0.shape[1:]
        # x0 = x0.view(self.bs, self.frame_len, c, h, w).transpose(1, 2)
        # # x0 = x0.transpose(0, 1).unsqueeze(0)
        # x0 = self.stf(x0)
        x1 = x1[(self.frame_len - 1)::self.frame_len]
        x2 = x2[(self.frame_len - 1)::self.frame_len]
        # x1, x2 = x1[-1, ...].unsqueeze(0), x2[-1, ...].unsqueeze(0)
        fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
        f_out0 = self.upsample(fpn_out0)  # 512/16
        f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16
        f_out0 = self.C3_p4(f_out0)  # 1024->512/16

        fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
        f_out1 = self.upsample(fpn_out1)  # 256/8
        f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
        pan_out2 = self.C3_p3(f_out1)  # 512->256/8

        p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
        p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
        pan_out1 = self.C3_n3(p_out1)  # 512->512/16

        p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
        p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
        pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32

        outputs = [pan_out2, pan_out1, pan_out0]
        return outputs

    def forward_neck(self, xs: Dict):
        features = [xs[f] for f in self.in_features]
        [x2, x1, x0] = features

        # aggregates the middle head
        # x0 = x0.transpose(0, 1).unsqueeze(0)
        # x0 = self.stf(x0)
        # x1, x2 = x1[-1, ...].unsqueeze(0), x2[-1, ...].unsqueeze(0)

        # aggregates the smallest head
        c, h, w = x0.shape[1:]
        x0 = x0.view(self.bs, self.frame_len, c, h, w).transpose(1, 2)
        # x0 = x0.transpose(0, 1).unsqueeze(0)
        x0 = self.stf(x0)
        x1 = x1[(self.frame_len - 1)::self.frame_len]
        x2 = x2[(self.frame_len - 1)::self.frame_len]
        # x1, x2 = x1[-1, ...].unsqueeze(0), x2[-1, ...].unsqueeze(0)
        fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
        f_out0 = self.upsample(fpn_out0)  # 512/16
        f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16
        f_out0 = self.C3_p4(f_out0)  # 1024->512/16

        fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
        f_out1 = self.upsample(fpn_out1)  # 256/8
        f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
        pan_out2 = self.C3_p3(f_out1)  # 512->256/8

        p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
        p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
        pan_out1 = self.C3_n3(p_out1)  # 512->512/16

        p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
        p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
        pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32

        outputs = [pan_out2, pan_out1, pan_out0]
        return outputs

    def forward(self, x):
        xs = self.forward_features(x)
        outputs = self.forward_neck(xs)

        return outputs


@registry.BACKBONES.register('resnet_50')
def resnet_50(cfg, pretrained=False):
    model = ResNet50(cfg, pretrained=pretrained)
    # if pretrained:
    #     model.load_state_dict(load_state_dict_from_url(model_urls['mobilenet_v3']), strict=False)
    return model


@registry.BACKBONES.register('resnet_pan')
def resnet_pan(cfg, pretrained=False):
    model = ResNetPAN(cfg, pretrained=pretrained)
    # if pretrained:
    #     model.load_state_dict(load_state_dict_from_url(model_urls['mobilenet_v3']), strict=False)
    return model


if __name__ == '__main__':
    from ssd.config import cfg
    from ssd.data.build import build_transforms, build_target_transform

    cfg.merge_from_file(
        "../../../configs/resnet/dark/resnet50_ssd320_dark_vid_large_head.yaml")
    cfg.freeze()
    model = ResNetPAN(cfg, False)
    dummy_input = torch.randn((3, 3, 320, 320))
    model(dummy_input)
