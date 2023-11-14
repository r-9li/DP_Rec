import torch
import torch.nn as nn
import torch.nn.functional as F
from model_lib.Res_1D import ResCNNStack


class DomainDiscriminator(nn.Module):

    def __init__(self, input_size, hidden_size, target_domain_num=1):
        super(DomainDiscriminator, self).__init__()
        kernel_size_list = []
        drop_rate_list = []
        use_sk_list = []
        use_cbam_list = []
        for i in range(2):
            kernel_size_list.append(3)
            drop_rate_list.append(0.15)
            use_sk_list.append(True)
            use_cbam_list.append(True)

        self.CNN_layer = ResCNNStack(input_size, hidden_size, 3, kernel_size_list, drop_rate_list, use_sk_list,
                                     use_cbam_list)

        self.pool = nn.AdaptiveMaxPool1d(1)
        self.bn = nn.BatchNorm1d(hidden_size, eps=1e-3)
        self.CNN_layer1 = nn.Conv1d(hidden_size, 1, 3)

    def forward(self, x):
        out = self.CNN_layer(x)
        out = self.bn(out)
        out = F.relu(out)
        out = self.CNN_layer1(out)
        out = self.pool(out)
        out = out.squeeze()
        out = out.unsqueeze(-1)
        return out  # BCELoss


if __name__ == "__main__":
    inputs = torch.randn(16, 128, 10000)
    sk = DomainDiscriminator(128, 64)
    out = sk(inputs)
    criterion = nn.L1Loss()
    loss = criterion(out, inputs)
    loss.backward()
    print(1)
