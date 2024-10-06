import torch
import torch.nn as nn


class ChannelAlign(nn.Module):
    def __init__(self, 
                 channel_list=[512, 1024, 512, 256, 256, 256]) -> None:
        super().__init__()
        self.conv_list = nn.ModuleList()

        for i, channel in enumerate(channel_list):
            self.conv_list.append(
                nn.Conv2d(channel, 256, kernel_size=(1, 1), stride=(1, 1))
            ) 

    def forward(self, features):
        new_features_list = []
        for i, feature in enumerate(features):
            new_features_list.append(self.conv_list[i](feature))

        return new_features_list
