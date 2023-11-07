import torch
import torch.nn as nn
import torch.nn.functional as F


class Concurrent(nn.Sequential):

    def __init__(self,
                 axis=1,
                 stack=False,
                 merge_type=None):
        super(Concurrent, self).__init__()
        assert (merge_type is None) or (merge_type in ["cat", "stack", "sum"])
        self.axis = axis
        if merge_type is not None:
            self.merge_type = merge_type
        else:
            self.merge_type = "stack" if stack else "cat"

    def forward(self, x):
        out = []
        for module in self._modules.values():
            out.append(module(x))
        if self.merge_type == "stack":
            out = torch.stack(tuple(out), dim=self.axis)
        elif self.merge_type == "cat":
            out = torch.cat(tuple(out), dim=self.axis)
        elif self.merge_type == "sum":
            out = torch.stack(tuple(out), dim=self.axis).sum(self.axis)
        else:
            raise NotImplementedError()
        return out


class SK_Net1D(nn.Module):
    def __init__(self, num_branches, input_channels, reduction=16,
                 min_channels=32):
        super(SK_Net1D, self).__init__()

        self.num_branches = num_branches
        self.input_channels = input_channels
        mid_channels = max(input_channels // reduction, min_channels)

        self.pool = nn.AdaptiveAvgPool1d(output_size=1)
        self.fc1 = torch.nn.Conv1d(input_channels, mid_channels, 1)
        self.fc2 = torch.nn.Conv1d(mid_channels, input_channels * num_branches, 1)
        self.softmax = nn.Softmax(dim=1)

        self.branches = Concurrent(stack=True)
        for i in range(num_branches):
            self.branches.add_module("branch{}".format(i), nn.Sequential(
                nn.Conv1d(in_channels=input_channels, out_channels=input_channels, kernel_size=3 + i * 2,
                          padding=1 + i),
                nn.BatchNorm1d(input_channels),
                nn.ReLU()))

    def forward(self, x):
        y = self.branches(x)

        u = y.sum(dim=1)
        s = self.pool(u)
        z = F.relu(self.fc1(s))
        w = self.fc2(z)

        batch = w.size(0)
        w = w.view(batch, self.num_branches, self.input_channels)
        w = self.softmax(w)
        w = w.unsqueeze(-1)

        y = y * w
        y = y.sum(dim=1)

        return y


if __name__ == "__main__":
    inputs = torch.randn(16, 128, 10000)
    sk = SK_Net1D(5, 128)
    out = sk(inputs)
    criterion = nn.L1Loss()
    loss = criterion(out, inputs)
    loss.backward()
    print(1)
