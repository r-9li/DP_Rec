import torch
import torch.nn as nn
from torchmetrics.functional.regression import pearson_corrcoef
import random


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


class Target_class_Loss(nn.Module):
    def __init__(self, alpha, class_num):
        super(Target_class_Loss, self).__init__()
        self.alpha = alpha
        self.class_num = class_num
        self.loss1 = nn.CrossEntropyLoss(reduction='none')
        self.loss2 = nn.CrossEntropyLoss(reduction='none')

    def forward(self, pred, target, batch_size, weight):
        random_target = []
        for i in range(batch_size):
            label_list = list(range(self.class_num))
            label_list.remove(target[i].tolist())
            random_target.append(random.choice(label_list))
        random_target = torch.tensor(random_target).cuda()
        Loss1 = self.loss1(pred, target)
        Loss2 = self.loss2(pred, random_target)
        Loss = torch.mean(Loss1 * weight) - self.alpha * torch.mean(Loss2 * weight)
        return Loss


if __name__ == "__main__":
    inputs1 = torch.randn(16, 256, 500)
    inputs2 = torch.randn(16, 256, 500)
    sk = DiffLoss()
    out = sk(inputs1, inputs2)
    criterion = nn.L1Loss()
    loss = criterion(out, inputs1)
    loss.backward()
    print(1)
