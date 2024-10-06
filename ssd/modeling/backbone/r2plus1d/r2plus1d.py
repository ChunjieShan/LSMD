import torch.nn as nn
from ssd.modeling import registry
from torch.nn.modules.utils import _triple
from typing import Union, Tuple, List

from .common import SpatioTemporalConv


class SpatioTemporalStemBlock(nn.Module):
    def __init__(self,
                 conv_type=SpatioTemporalConv,
                 in_channels: int = None,
                 out_channels: int = None,
                 kernel_size: Union[int, List] = None,
                 padding: Union[int, List] = None,
                 down_sample: bool = False):
        super(SpatioTemporalStemBlock, self).__init__()

        self.down_sample = down_sample

        if down_sample:
            conv1 = conv_type(in_channels, out_channels, kernel_size, padding=padding, stride=(1, 2, 2))
            self.down_sample_conv = SpatioTemporalConv(in_channels, out_channels, 1, stride=(1, 2, 2))
            self.down_sample_bn = nn.BatchNorm3d(out_channels)
        else:
            conv1 = conv_type(in_channels, out_channels, kernel_size, padding=padding)

        bn1 = nn.BatchNorm3d(out_channels)

        conv2 = conv_type(out_channels, out_channels, kernel_size, padding=padding)
        bn2 = nn.BatchNorm3d(out_channels)

        self.relu = nn.ReLU()

        self.stem = nn.Sequential(*[
            conv1, bn1, self.relu,
            conv2, bn2
        ])

    def forward(self, x):
        out = self.stem(x)

        if self.down_sample:
            x = self.down_sample_bn(self.down_sample_conv(x))
            return self.relu(x + out)

        return self.relu(x + out)


class SpatioTemporalResBlock(nn.Module):
    r"""Single block for the ResNet network. Uses SpatioTemporalConv in
        the standard ResNet block layout (conv->batchnorm->ReLU->conv->batchnorm->sum->ReLU)

        Args:
            in_channels (int): Number of channels in the input tensor.
            out_channels (int): Number of channels in the output produced by the block.
            kernel_size (int or tuple): Size of the convolving kernels.
            down_sample (bool, optional): If ``True``, the output size is to be smaller than the input. Default: ``False``
        """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stem_type=SpatioTemporalStemBlock,
                 layer_size=1,
                 down_sample=True,
                 allow_padding=True):
        super(SpatioTemporalResBlock, self).__init__()

        # If down_sample == True, the first conv of the layer has stride = 2
        # to halve the residual output size, and the input x is passed
        # through a separable 1x1x1 conv with stride = 2 to also halve it.

        # no pooling layers are used inside ResNet
        self.down_sample_conv = SpatioTemporalConv(in_channels, out_channels, 1, stride=(1, 2, 2))
        self.down_sample_bn = nn.BatchNorm3d(out_channels)

        # to allow for SAME padding
        if allow_padding:
            if isinstance(kernel_size, int):
                padding = kernel_size // 2
            else:
                padding = []
                for size in kernel_size:
                    padding.append(size // 2)

                padding = tuple(padding)
        else:
            padding = (1, 0, 0)

        self.blocks = nn.ModuleList([])
        self.blocks.append(
            stem_type(conv_type=SpatioTemporalConv,
                      in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      padding=padding,
                      down_sample=down_sample)
        )

        for _ in range(layer_size - 1):
            self.blocks.append(
                stem_type(conv_type=SpatioTemporalConv,
                          in_channels=out_channels,
                          out_channels=out_channels,
                          kernel_size=kernel_size,
                          padding=padding,
                          down_sample=False)
            )

    def forward(self, x):
        for i, module in enumerate(self.blocks):
            x = module(x)

        return x


class R2Plus1DNet(nn.Module):
    r"""Forms the overall ResNet feature extractor by initializng 5 layers, with the number of blocks in
    each layer set by layer_sizes, and by performing a global average pool at the end producing a
    512-dimensional vector for each element in the batch.

        Args:
            layer_sizes (tuple): An iterable containing the number of blocks in each layer
            block_type (Module, optional): Type of block that is to be used to form the layers. Default: SpatioTemporalResBlock.
        """

    def __init__(self, layer_sizes, block_type=SpatioTemporalResBlock):
        super(R2Plus1DNet, self).__init__()
        self.conv1 = nn.Sequential(*[
            SpatioTemporalConv(3, 64, (5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        ])
        # output of conv2 is same size as of conv1, no downsampling needed. kernel_size 3x3x3
        self.conv2 = block_type(64, 64, (3, 3, 3), layer_size=layer_sizes[0], down_sample=False)
        self.conv3 = block_type(64, 128, (3, 3, 3), layer_size=layer_sizes[1], down_sample=True)
        self.conv4 = block_type(128, 256, (3, 3, 3), layer_size=layer_sizes[2], down_sample=True)
        self.conv5 = block_type(256, 512, (3, 3, 3), layer_size=layer_sizes[3], down_sample=True)
        self.conv6 = block_type(512, 256, (3, 3, 3), layer_size=1, down_sample=True)
        self.conv7 = block_type(256, 256, (3, 3, 3), layer_size=1, down_sample=True)
        # self.conv8 = block_type(256, 256, (13, 3, 3), layer_size=3, down_sample=True)

        # self.conv6 = SpatioTemporalConv(in_channels=512,
        #                                 out_channels=256,
        #                                 kernel_size=(13, 3, 3),
        #                                 stride=(1, 2, 2),
        #                                 padding=(6, 1, 1),
        #                                 is_relu=False)
        #
        # self.conv7 = SpatioTemporalConv(in_channels=256,
        #                                 out_channels=256,
        #                                 kernel_size=(13, 3, 3),
        #                                 stride=(1, 2, 2),
        #                                 padding=(6, 1, 1),
        #                                 is_relu=False)
        #
        self.conv8 = SpatioTemporalConv(in_channels=256,
                                        out_channels=256,
                                        kernel_size=(3, 3, 3),
                                        stride=(1, 2, 2),
                                        padding=(1, 0, 0),
                                        is_relu=False)

    def forward(self, x):
        features = []
        # 64 75 75
        x = self.conv1(x)
        # 64 75 75
        x = self.conv2(x)
        # 128 38 38
        x = self.conv3(x)
        features.append(x.squeeze(0).permute(1, 0, 2, 3))
        # 256 19 19
        x = self.conv4(x)
        features.append(x.squeeze(0).permute(1, 0, 2, 3))
        # 512 10 10
        x = self.conv5(x)
        features.append(x.squeeze(0).permute(1, 0, 2, 3))
        # 256 5 5
        x = self.conv6(x)
        features.append(x.squeeze(0).permute(1, 0, 2, 3))
        # 256 3 3
        x = self.conv7(x)
        features.append(x.squeeze(0).permute(1, 0, 2, 3))
        # # 256 1 1
        x = self.conv8(x)
        features.append(x.squeeze(0).permute(1, 0, 2, 3))

        return features


@registry.BACKBONES.register('r2plus1d')
def r2plus1d(cfg, pretrained=True):
    model = R2Plus1DNet(block_type=SpatioTemporalResBlock, layer_sizes=(2, 2, 2, 2))

    return model


if __name__ == '__main__':
    import torch
    import onnx

    # block = SpatioTemporalResBlock(64, 64, 3, layer_size=3)

    model = R2Plus1DNet(block_type=SpatioTemporalResBlock, layer_sizes=(3, 4, 6, 3))
    # model = R2Plus1DNet(layer_sizes=(2, 2, 1, 1)).to("cuda")
    dummy_input = torch.randn((1, 3, 32, 300, 300))
    print(model)
    print(model(dummy_input))

    # print("Exporting to ...")
    # torch.onnx.export(model,
    #                   dummy_input,
    #                   f="./r2plus1d.onnx",
    #                   opset_version=11)

    # onnx.save(onnx.shape_inference.infer_shapes(onnx.load("./r2plus1d.onnx")), "./r2plus1d.onnx")
    # model(dummy_input)
