"""
modules.py - This file stores the rathering boring network blocks.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from mmcv.cnn import ConvModule


class KeyProjection(nn.Module):
    def __init__(self, in_channels, out_channels, need_s=False, need_e=False):
        super().__init__()
        self.need_s = need_s
        self.need_e = need_e

        self.key_proj = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        # shrinkage
        self.d_proj = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)
        # selection
        self.e_proj = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        nn.init.orthogonal_(self.key_proj.weight.data)
        nn.init.zeros_(self.key_proj.bias.data)

    def forward(self, x):
        if self.need_e or self.need_s:
            shrinkage = self.d_proj(x) ** 2 + 1
            selection = torch.sigmoid(self.e_proj(x))

            return self.key_proj(x), shrinkage, selection

        return self.key_proj(x), None, None


class SequenceConv(nn.ModuleList):
    """Sequence conv module.

    Args:
        in_channels (int): input tensor channel.
        out_channels (int): output tensor channel.
        kernel_size (int): convolution kernel size.
        sequence_num (int): sequence length.
        conv_cfg (dict): convolution config dictionary.
        norm_cfg (dict): normalization config dictionary.
        act_cfg (dict): activation config dictionary.
    """

    def __init__(self, in_channels, out_channels, kernel_size, sequence_num, conv_cfg, norm_cfg, act_cfg):
        super(SequenceConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sequence_num = sequence_num
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        for _ in range(sequence_num):
            self.append(
                ConvModule(
                    self.in_channels,
                    self.out_channels,
                    self.kernel_size,
                    padding=self.kernel_size // 2,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg)
            )

    def forward(self, sequence_imgs):
        """

        Args:
            sequence_imgs (Tensor): TxBxCxHxW

        Returns:
            sequence conv output: TxBxCxHxW
        """
        sequence_outs = []
        assert sequence_imgs.shape[0] == self.sequence_num
        for i, sequence_conv in enumerate(self):
            sequence_out = sequence_conv(sequence_imgs[i, ...])
            sequence_out = sequence_out.unsqueeze(0)
            sequence_outs.append(sequence_out)

        sequence_outs = torch.cat(sequence_outs, dim=0)  # TxBxCxHxW
        return sequence_outs


class MemoryModule(nn.Module):
    """Memory read module.
    Args:

    """

    def __init__(self,
                 matmul_norm=False):
        super(MemoryModule, self).__init__()
        self.matmul_norm = matmul_norm

    def forward(self, memory_keys, memory_values, query_key, query_value):
        """
        Memory Module forward.
        Args:
            memory_keys (Tensor): memory keys tensor, shape: TxBxCxHxW
            memory_values (Tensor): memory values tensor, shape: TxBxCxHxW
            query_key (Tensor): query keys tensor, shape: BxCxHxW
            query_value (Tensor): query values tensor, shape: BxCxHxW

        Returns:
            Concat query and memory tensor.
        """
        sequence_num, batch_size, key_channels, height, width = memory_keys.shape
        _, _, value_channels, _, _ = memory_values.shape
        assert query_key.shape[1] == key_channels and query_value.shape[1] == value_channels
        memory_keys = memory_keys.permute(1, 2, 0, 3, 4).contiguous()  # BxCxTxHxW
        memory_keys = memory_keys.view(batch_size, key_channels, sequence_num * height * width)  # BxCxT*H*W

        query_key = query_key.view(batch_size, key_channels, height * width).permute(0, 2, 1).contiguous()  # BxH*WxCk
        key_attention = torch.bmm(query_key, memory_keys)  # BxH*WxT*H*W
        if self.matmul_norm:
            key_attention = (key_channels ** -.5) * key_attention
        key_attention = F.softmax(key_attention, dim=-1)  # BxH*WxT*H*W

        memory_values = memory_values.permute(1, 2, 0, 3, 4).contiguous()  # BxCxTxHxW
        memory_values = memory_values.view(batch_size, value_channels, sequence_num * height * width)
        memory_values = memory_values.permute(0, 2, 1).contiguous()  # BxT*H*WxC
        memory = torch.bmm(key_attention, memory_values)  # BxH*WxC
        memory = memory.permute(0, 2, 1).contiguous()  # BxCxH*W
        memory = memory.view(batch_size, value_channels, height, width)  # BxCxHxW

        query_memory = torch.cat([query_value, memory], dim=1)
        return query_memory
