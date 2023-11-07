import torch
import torch.nn as nn
import torch.nn.functional as F
from model_lib.Res_1D import ResCNNStack

Encoder1D_Param = {'channel_list': [64, 128, 256, 512],
                   'DownSample_kernel_size_list': [3, 3, 3, 3],
                   'kernel_size_list': [[3, 3, 3, 3],
                                        [3, 3, 3],
                                        [3, 3, 3],
                                        [3, 3, 3]],
                   'drop_rate_list': [[0., 0., 0., 0.],
                                      [0.2, 0.1, 0.1],
                                      [0.3, 0.2, 0.1],
                                      [0.3, 0.2, 0.1]],
                   'use_sk_list': [[True, True, True, False],
                                   [True, False, False],
                                   [True, False, False],
                                   [False, False, False]],
                   'use_cbam_list': [[False, False, True, True],
                                     [False, False, True],
                                     [False, False, True],
                                     [True, True, True]]}


class Encoder(nn.Module):
    """
    Encoder stack
    """

    def __init__(self, input_channel, channel_list, DownSample_kernel_size_list, kernel_size_list, drop_rate_list,
                 use_sk_list, use_cbam_list):
        super().__init__()

        self.conv1 = nn.Conv1d(input_channel, 32, 7, stride=2, padding=7 // 2)
        self.norm = nn.BatchNorm1d(32)

        last_channel_size = 32
        self.members = nn.ModuleList()
        for channel_size, DownSample_kernel_size, kernel_size, drop_rate, use_sk, use_cbam in zip(channel_list,
                                                                                                  DownSample_kernel_size_list,
                                                                                                  kernel_size_list,
                                                                                                  drop_rate_list,
                                                                                                  use_sk_list,
                                                                                                  use_cbam_list):
            self.members.append(
                ResCNNStack(last_channel_size, channel_size, DownSample_kernel_size, kernel_size, drop_rate, use_sk,
                            use_cbam))
            last_channel_size = channel_size

    def forward(self, x):
        y = self.conv1(x)
        y = self.norm(y)
        y = F.relu(y)

        for member in self.members:
            y = member(y)

        return y


if __name__ == "__main__":
    inputs = torch.randn(16, 1, 8192)
    sk = Encoder(input_channel=1, channel_list=Encoder1D_Param['channel_list'],
                 DownSample_kernel_size_list=Encoder1D_Param['DownSample_kernel_size_list'],
                 kernel_size_list=Encoder1D_Param['kernel_size_list'], drop_rate_list=Encoder1D_Param['drop_rate_list'],
                 use_sk_list=Encoder1D_Param['use_sk_list'], use_cbam_list=Encoder1D_Param['use_cbam_list'])
    out = sk(inputs)
    criterion = nn.L1Loss()
    loss = criterion(out, inputs)
    loss.backward()
    print(1)
