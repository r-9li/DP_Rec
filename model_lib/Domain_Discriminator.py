import torch
import torch.nn as nn
import torch.nn.functional as F


class DomainDiscriminator(nn.Module):

    def __init__(self, input_size, hidden_size, target_domain_num=1):
        super(DomainDiscriminator, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool1d(2))
        self.layer2 = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool1d(2))
        self.layer3 = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool1d(2))

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.final_norm = nn.BatchNorm1d(hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, target_domain_num + 1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.pool(out)
        out = out.squeeze()
        out = self.final_norm(F.relu(self.fc1(out)))
        out = self.fc2(out)
        return out  # CELoss


if __name__ == "__main__":
    inputs = torch.randn(16, 128, 10000)
    sk = DomainDiscriminator(128, 64)
    out = sk(inputs)
    criterion = nn.L1Loss()
    loss = criterion(out, inputs)
    loss.backward()
    print(1)
