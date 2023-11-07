import torch
import torch.nn as nn
import torch.nn.functional as F


class DP_Classifier(nn.Module):
    def __init__(self, classes_number, hidden_size=128):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.norm = nn.BatchNorm1d(hidden_size)
        self.fc = nn.Linear(hidden_size, hidden_size // 2)

        self.final_norm = nn.BatchNorm1d(hidden_size // 2)
        self.final_fc = nn.Linear(hidden_size // 2, classes_number)

    def forward(self, x):
        y = self.pool(x)
        y = y.squeeze()
        if x.size()[0] == 1:
            y = y.unsqueeze(0)

        y = self.norm(y)
        y = F.relu(y)
        y = self.fc(y)

        y = self.final_norm(y)
        y = F.relu(y)
        y = self.final_fc(y)

        return y


if __name__ == "__main__":
    inputs = torch.randn(16, 128, 10000)
    sk = DP_Classifier(4)
    out = sk(inputs)
    criterion = nn.L1Loss()
    loss = criterion(out, inputs)
    loss.backward()
    print(1)
