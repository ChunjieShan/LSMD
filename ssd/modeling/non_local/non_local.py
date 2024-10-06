import torch
from torch import nn
from torch.nn import functional as F


class _NonLocalBlockND(nn.Module):
    def __init__(self,
                 in_channels,
                 inter_channels=None,
                 dimension=3,
                 sub_sample=True,
                 bn_layer=True,
                 return_4d=True,
                 need_det_token=False):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.return_4d = return_4d
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        self.need_det_token = need_det_token
        if self.need_det_token:
            self.det_token = nn.Parameter(torch.randn(1, 1024, 1, 20, 20))
            nn.init.trunc_normal_(self.det_token, std=0.02)
            self.apply(_init_vit_weights)

        self.pre_det_logits = nn.Sequential(
            nn.Conv3d(1024, 1024, (1, 3, 3), (1, 1, 1), padding=(0, 1, 1)),
            nn.ReLU(),
        )

        # else:
        #     self.pre_det_logits = nn.Identity()

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        if x.dim() == 4:
            x = x.transpose(0, 1).unsqueeze(0)

        if self.need_det_token:
            x = torch.cat([self.det_token, x], dim=2)
        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        # if self.need_det_token:
        #     z = z[:, :, -1, :, :].unsqueeze(2)
        #     z = self.pre_det_logits(z)
        # else:
        #     z = z[:, :, -1, :, :].unsqueeze(2)
        #     z = self.pre_det_logits(z)

        if self.return_4d:
            return z.squeeze(0).transpose(0, 1)

        return z


class NONLocalBlock1D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock1D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=1, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class NONLocalBlock2D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock2D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=2, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class NONLocalBlock3D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True, return_4d=True, need_det_token=False):
        super(NONLocalBlock3D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=3, sub_sample=sub_sample,
                                              bn_layer=bn_layer, return_4d=return_4d,
                                              need_det_token=need_det_token)


class NonLocalPair(nn.Module):
    def __init__(self, in_channels):
        super(NonLocalPair, self).__init__()
        # nl_conv_pair = [
        #     NONLocalBlock3D(in_channels=in_channels),
        #     nn.Conv2d(in_channels, in_channels * 2, (1, 1), (1, 1)),
        #     NONLocalBlock3D(in_channels=in_channels * 2),
        #     nn.Conv2d(in_channels * 2, in_channels * 2, (1, 1), (1, 1)),
        #     NONLocalBlock3D(in_channels=in_channels * 2),
        #     nn.Conv2d(in_channels * 2, in_channels, (1, 1), (1, 1)),
        # ]
        self.nl_block = nn.Sequential(
            NONLocalBlock3D(in_channels=in_channels),
            nn.Conv2d(in_channels, in_channels * 2, (1, 1), (1, 1)),
            NONLocalBlock3D(in_channels=in_channels * 2),
            nn.Conv2d(in_channels * 2, in_channels * 2, (1, 1), (1, 1)),
            NONLocalBlock3D(in_channels=in_channels * 2),
            nn.Conv2d(in_channels * 2, in_channels, (1, 1), (1, 1)),
        )

    def forward(self, x):
        return self.nl_block(x)


def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


if __name__ == '__main__':
    import torch

    for (sub_sample, bn_layer) in [(True, True), (False, False), (True, False), (False, True)]:
        # img = torch.zeros(2, 3, 20)
        # net = NONLocalBlock1D(3, sub_sample=sub_sample, bn_layer=bn_layer)
        # out = net(img)
        # print(out.size())
        #
        # img = torch.zeros(2, 3, 20, 20)
        # net = NONLocalBlock2D(3, sub_sample=sub_sample, bn_layer=bn_layer)
        # out = net(img)
        # print(out.size())

        img = torch.randn(1, 64, 16, 20, 20)
        net = NONLocalBlock3D(64, sub_sample=sub_sample, bn_layer=bn_layer)
        out = net(img)
        print(out.size())

