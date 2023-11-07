import torch
import torch.nn as nn
import torch.nn.functional as F

from model_lib.Conv1D_Encoder import Encoder, Encoder1D_Param
from model_lib.Transformer import Transformer


class DP_Feature_Encoder(nn.Module):
    def __init__(self, input_size, Encoder_Param, drop_rate=0.1,
                 original_compatible="non-conservative", hidden_size=128):
        super().__init__()
        self.Encoder = Encoder(input_size, channel_list=Encoder_Param['channel_list'],
                               DownSample_kernel_size_list=Encoder_Param['DownSample_kernel_size_list'],
                               kernel_size_list=Encoder_Param['kernel_size_list'],
                               drop_rate_list=Encoder_Param['drop_rate_list'],
                               use_sk_list=Encoder_Param['use_sk_list'], use_cbam_list=Encoder_Param['use_cbam_list'])

        if original_compatible == 'non-conservative':
            eps = 1e-7  # See Issue #96 - original models use tensorflow default epsilon of 1e-7
        else:
            eps = 1e-5

        self.conv = nn.Conv1d(Encoder_Param['channel_list'][-1], hidden_size, 1)
        self.norm = nn.BatchNorm1d(hidden_size)

        self.transformer1 = Transformer(input_size=hidden_size, drop_rate=drop_rate, eps=eps)
        self.transformer2 = Transformer(input_size=hidden_size, drop_rate=drop_rate, eps=eps)
        self.transformer3 = Transformer(input_size=hidden_size, drop_rate=drop_rate, eps=eps)

    def forward(self, x):
        y = self.Encoder(x)

        y = self.conv(y)
        y = self.norm(y)
        y = F.relu(y)

        y, _ = self.transformer1(y)
        y, _ = self.transformer2(y)
        y, _ = self.transformer3(y)

        return y


if __name__ == "__main__":
    inputs = torch.randn(16, 1, 10000)
    sk = DP_Feature_Encoder(1, Encoder1D_Param)
    out = sk(inputs)
    criterion = nn.L1Loss()
    loss = criterion(out, inputs)
    loss.backward()
    print(1)
