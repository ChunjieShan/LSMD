import torch
import torch.nn as nn
from torchvision.ops import MultiScaleRoIAlign


class SSDRPN(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.image_shape = [(cfg.INPUT.IMAGE_SIZE, cfg.INPUT.IMAGE_SIZE)]
        # self.image_shape = [image_shape for _ in range(32)]
        self.roi_align = MultiScaleRoIAlign(featmap_names=["0"],
                                            output_size=7,
                                            sampling_ratio=2)

        self.reduce_dim = nn.Conv2d(in_channels=1024,
                                    out_channels=512,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1))

    def forward(self, features, proposals):
        features["0"] = self.reduce_dim(features["0"])
        object_feat = self.roi_align(features, proposals, self.image_shape)

        return object_feat
        # return self.reduce_dim(object_feat)
