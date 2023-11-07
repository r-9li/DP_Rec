'''
    References
    ----------
    [1](http://arxiv.org/abs/1807.06521v2)
        @InProceedings{Woo_2018_ECCV,
            author = {Woo, Sanghyun and Park, Jongchan and Lee, Joon-Young and Kweon, In So},
            title = {CBAM: Convolutional Block Attention Module},
            booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
            month = {September},
            year = {2018}
        }
'''

# Code:   https://github.com/Jongchan/attention-module/blob/master/MODELS/cbam.py
# Note:   Jiaqili@zju.edu.cn modified it as 1d-CBAM

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["CBAM_1D"]


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm1d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool1d(x, x.size(2), stride=x.size(2))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool1d(x, x.size(2), stride=x.size(2))
                channel_att_raw = self.mlp(max_pool)
            # elif pool_type=='lp':
            #     lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
            #     channel_att_raw = self.mlp( lp_pool )
            elif pool_type == 'lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).expand_as(x)
        return x * scale


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)
        return x * scale


class CBAM_1D(nn.Module):
    '''CBAM: Convolutional Block Attention Module (1d) in NvTK.

    Parameters
    ----------
    gate_channels : int
        Number of gate channels
    reduction_ratio : int, optional
        Number of reduction ratio in ChannelGate
    pool_types : list of str, optional
        List of Pooling types in ChannelGate, Default is ['avg', 'max'].
        Should be in the range of `['avg', 'max', 'lse']`
        (e.g. `pool_types=['avg', 'max', 'lse']`)
    no_spatial : bool, optional
        Whether not to use SpatialGate, Default is False.

    Attributes
    ----------
    no_spatial : bool

    ChannelGate : nn.Module
        The Channel Gate Module in CBAM
    SpatialGate : nn.Module
        The Spatial Gate Module in CBAM
    attention : nn.Tensor
        The overall attention weights

    Tensor flows
    ----------
    -> ChannelGate(x)

    -> SpatialGate(x_out)(x) if not no_spatial

    '''

    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM_1D, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
        self.attention = None

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)

        self.attention = x_out / x  # (Batch, Filter, Seq_len)
        return x_out

    def get_attention(self):
        '''return the attention weights in a batch'''
        return self.attention

if __name__ == "__main__":
    inputs = torch.randn(16, 128, 10000)
    sk = CBAM_1D(128)
    out = sk(inputs)
    criterion = nn.L1Loss()
    loss = criterion(out, inputs)
    loss.backward()
    print(1)
