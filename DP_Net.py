import torch
import torch.nn as nn

from model_lib.Classifier import DP_Classifier
from model_lib.Conv1D_Encoder import Encoder1D_Param
from model_lib.Feature_Encoder import DP_Feature_Encoder


class DP_Net(nn.Module):
    def __init__(self, classes_number, input_size, Encoder_Param, drop_rate=0.1,
                 original_compatible="non-conservative", hidden_size=128):
        super().__init__()
        self.Feature_Encoder = DP_Feature_Encoder(input_size, Encoder_Param, drop_rate, original_compatible,
                                                  hidden_size)
        self.Classifier = DP_Classifier(classes_number, hidden_size)

    def forward(self, x):
        y = self.Feature_Encoder(x)

        y = self.Classifier(y)

        return y


if __name__ == "__main__":
    inputs = torch.randn(16, 1, 10000)
    sk = DP_Net(4, 1, Encoder1D_Param)
    out = sk(inputs)
    criterion = nn.L1Loss()
    loss = criterion(out, inputs)
    loss.backward()
    print(1)
