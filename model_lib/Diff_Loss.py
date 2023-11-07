import torch
import torch.nn as nn
from torchmetrics.functional.regression import pearson_corrcoef


class DiffLoss(nn.Module):

    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, input1, input2):
        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1).t()
        input2 = input2.view(batch_size, -1).t()
        diff_loss = pearson_corrcoef(input1, input2)
        diff_loss = torch.mean(torch.abs(diff_loss))

        return diff_loss


if __name__ == "__main__":
    inputs1 = torch.randn(16, 256, 500)
    inputs2 = torch.randn(16, 256, 500)
    sk = DiffLoss()
    out = sk(inputs1, inputs2)
    criterion = nn.L1Loss()
    loss = criterion(out, inputs1)
    loss.backward()
    print(1)
