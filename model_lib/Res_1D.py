import torch
import torch.nn as nn
import torch.nn.functional as F
from model_lib.SKNet_1D import SK_Net1D
from model_lib.CBAM_1D import CBAM_1D


class ResCNNStack(nn.Module):
    def __init__(self, InputChannle_size, OutputChannle_size, DownSample_kernel_size, kernel_size_list, drop_rate_list,
                 use_sk_list, use_cbam_list):
        super().__init__()

        self.DownSample_Block = ResCNNBlock_DownSample(InputChannle_size, OutputChannle_size, DownSample_kernel_size)
        self.members = nn.ModuleList()
        for kernel_size, drop_rate, use_sk, use_cbam in zip(kernel_size_list, drop_rate_list, use_sk_list,
                                                            use_cbam_list):
            self.members.append(
                ResCNNBlock_Normal(Channle_size=OutputChannle_size, kernel_size=kernel_size, drop_rate=drop_rate,
                                   use_sk=use_sk, use_cbam=use_cbam))

    def forward(self, x):
        y = self.DownSample_Block(x)
        for member in self.members:
            y = member(y)

        return y


class ResCNNBlock_DownSample(nn.Module):
    def __init__(self, InputChannle_size, OutputChannle_size, kernel_size):
        super().__init__()

        self.conv1 = nn.Conv1d(InputChannle_size, InputChannle_size // 2, kernel_size=1)
        self.norm1 = nn.BatchNorm1d(InputChannle_size // 2, eps=1e-3)

        self.conv2 = nn.Conv1d(InputChannle_size // 2, InputChannle_size // 2, kernel_size=kernel_size,
                               padding=kernel_size // 2,
                               stride=2)
        self.norm2 = nn.BatchNorm1d(InputChannle_size // 2, eps=1e-3)

        self.conv3 = nn.Conv1d(InputChannle_size // 2, OutputChannle_size, kernel_size=1)

        self.conv4 = nn.Conv1d(InputChannle_size, OutputChannle_size, kernel_size=1, stride=2)

    def forward(self, x):
        y = self.conv1(x)
        y = self.norm1(y)
        y = F.relu(y)

        y = self.conv2(y)
        y = self.norm2(y)
        y = F.relu(y)

        y = self.conv3(y)

        res = self.conv4(x)

        y += res
        return y


class ResCNNBlock_Normal(nn.Module):
    def __init__(self, Channle_size, kernel_size, drop_rate=0, use_sk=False, use_cbam=False):
        super().__init__()

        if drop_rate > 0:
            self.dropout = nn.Dropout1d(drop_rate)

        self.drop_rate = drop_rate
        self.use_cbam = use_cbam

        self.conv1 = nn.Conv1d(Channle_size, Channle_size // 2, 1)
        self.norm1 = nn.BatchNorm1d(Channle_size // 2, eps=1e-3)

        if use_sk:
            self.conv2 = SK_Net1D(5, Channle_size // 2)
        else:
            self.conv2 = nn.Conv1d(Channle_size // 2, Channle_size // 2, kernel_size, padding=kernel_size // 2)

        self.norm2 = nn.BatchNorm1d(Channle_size // 2, eps=1e-3)

        self.conv3 = nn.Conv1d(Channle_size // 2, Channle_size, 1)

        if use_cbam:
            self.cbam = CBAM_1D(Channle_size)

    def forward(self, x):
        if self.drop_rate > 0:
            y = self.dropout(x)
        else:
            y = x
        y = self.conv1(y)
        y = self.norm1(y)
        y = F.relu(y)

        if self.drop_rate > 0:
            y = self.dropout(y)
        y = self.conv2(y)
        y = self.norm2(y)
        y = F.relu(y)

        y = self.conv3(y)

        if self.use_cbam:
            y = self.cbam(y)

        y += x

        return x + y


if __name__ == "__main__":
    inputs = torch.randn(16, 128, 10000)
    sk = ResCNNStack(128, 256, 3, [3, 3, 3, 3, 3], [0.1, 0.1, 0.1, 0.1, 0.1], [True, True, True, True, True],
                     [True, True, True, True, True])
    out = sk(inputs)
    criterion = nn.L1Loss()
    loss = criterion(out, inputs)
    loss.backward()
    print(1)
