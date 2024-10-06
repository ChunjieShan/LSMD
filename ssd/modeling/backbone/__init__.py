from ssd.modeling import registry
from .darknet import darknet_stam_pan, darknet_stf_pan
from .efficient_net import EfficientNet
from .mobilenet import MobileNetV2
from .mobilenetv3 import MobileNetV3
from .resnet import resnet_50, resnet_pan
from .vgg import VGG

__all__ = ['build_backbone', 'VGG', 'MobileNetV2', 'EfficientNet', 'MobileNetV3', 'resnet_50', 'resnet_pan',
           'darknet_stam_pan', 'darknet_stf_pan']


def build_backbone(cfg):
    return registry.BACKBONES[cfg.MODEL.BACKBONE.NAME](cfg, cfg.MODEL.BACKBONE.PRETRAINED)
