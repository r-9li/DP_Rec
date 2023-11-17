import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from DP_Net import DP_Net
from DP_Net_CrossDomain import DP_Net_CrossDomain
from model_lib.Diff_Loss import DiffLoss, Target_class_Loss


def accuracy(output, label):
    batch_size = output.size(0)
    _, pred = torch.topk(output, k=1, dim=1, largest=True, sorted=True)
    pred = pred.T
    correct = torch.eq(pred, label.contiguous().view(1, -1).expand_as(pred))
    correct_1 = correct[:1].contiguous().view(-1).float().sum(dim=0, keepdim=True)
    res = correct_1 * 100 / batch_size
    res = float(res)

    return res


class DP_Net_Lightning(pl.LightningModule):
    def __init__(self, classes_number, input_size, Encoder_Param, drop_rate=0.1,
                 original_compatible="non-conservative", hidden_size=128, acc_check_step=100):
        super().__init__()
        self.network = DP_Net(classes_number=classes_number, input_size=input_size, Encoder_Param=Encoder_Param,
                              drop_rate=drop_rate, original_compatible=original_compatible,
                              hidden_size=hidden_size)

        self.loss = nn.CrossEntropyLoss()
        self.acc_check_step = acc_check_step

        self.val_acc = []
        self.test_acc = []

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self.network(x)
        Loss = self.loss(output, y)

        self.log('Loss', Loss, prog_bar=True)
        if batch_idx % self.acc_check_step == 0:
            output = F.softmax(output, dim=1)
            self.log('train_acc', accuracy(output, y), prog_bar=True)

        return Loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self.network(x)
        output = F.softmax(output, dim=1)
        acc = accuracy(output, y)
        self.val_acc.append(acc)
        self.log('val_step_acc', acc, prog_bar=True)

    def on_validation_epoch_end(self):
        final_acc = sum(self.val_acc) / len(self.val_acc)
        self.val_acc.clear()
        self.log('val_final_acc', final_acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        output = self.network(x)
        output = F.softmax(output, dim=1)
        acc = accuracy(output, y)
        self.test_acc.append(acc)
        self.log('test_step_acc', acc, prog_bar=True)

    def on_test_epoch_end(self):
        final_acc = sum(self.test_acc) / len(self.test_acc)
        self.test_acc.clear()
        self.log('test_final_acc', final_acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer


class DP_Net_CrossDomain_Lightning(pl.LightningModule):
    def __init__(self, classes_number, input_size, Encoder_Param, domain_loss_weight, diff_loss_weight,
                 target_loss_weight, Use_Spurious_Label, drop_rate=0.1,
                 original_compatible="non-conservative", hidden_size=128, discriminator_hidden_size=64,
                 target_domain_num=1, acc_check_step=100):
        super().__init__()
        self.Use_Spurious_Label = Use_Spurious_Label
        self.network = DP_Net_CrossDomain(classes_number, input_size, Encoder_Param, Use_Spurious_Label, drop_rate,
                                          original_compatible,
                                          hidden_size, discriminator_hidden_size, target_domain_num)

        self.class_loss = nn.CrossEntropyLoss()
        self.target_class_loss = Target_class_Loss(0.6, classes_number)
        self.domain_loss = nn.BCEWithLogitsLoss()
        self.diff_loss = DiffLoss()

        self.domain_loss_weight = domain_loss_weight
        self.diff_loss_weight = diff_loss_weight
        self.target_loss_weight = target_loss_weight

        self.acc_check_step = acc_check_step

        self.val_acc_source = []
        self.val_acc_target = []

        self.test_acc_source = []
        self.test_acc_target = []

        self.save_hyperparameters()

    def init_subnetwork(self, weight_path, additional_init=False):
        weight = torch.load(weight_path)['state_dict']
        feature_encoder_dict = {}
        classifier_dict = {}
        for k, v in weight.items():
            if k.split('.')[1] == 'Feature_Encoder':
                k_now = k[24:]
                feature_encoder_dict[k_now] = v
            elif k.split('.')[1] == 'Classifier':
                k_now = k[19:]
                classifier_dict[k_now] = v

        self.network.Feature_Encoder_Share.load_state_dict(feature_encoder_dict)
        self.network.Classifier.load_state_dict(classifier_dict)
        if additional_init:
            self.network.Feature_Encoder_Source.load_state_dict(feature_encoder_dict)
            self.network.Feature_Encoder_Target.load_state_dict(feature_encoder_dict)

    def training_step(self, batch, batch_idx):
        source, target, label, target_loss_weight = batch
        batch_size = source.size(0)
        alpha = 1.

        (feature_share_target, feature_share_source, feature_private_target, feature_private_source, pred_class_label,
         pred_domain_label_share_source, pred_domain_label_share_target, pred_domain_label_private_source,
         pred_domain_label_private_target, pred_class_label_target) = self.network(
            (source, target, alpha))

        Loss_cls = self.class_loss(pred_class_label, label)
        if self.Use_Spurious_Label:
            Loss_cls_target = self.target_class_loss(pred_class_label_target, label, batch_size, target_loss_weight)

        Loss_domain_share_source = self.domain_loss(pred_domain_label_share_source,
                                                    torch.zeros(
                                                        (batch_size, 1)).cuda())  # Source Domain 0 Target Domain 1
        Loss_domain_private_source = self.domain_loss(pred_domain_label_private_source,
                                                      torch.zeros((batch_size, 1)).cuda())
        Loss_domain_share_target = self.domain_loss(pred_domain_label_share_target,
                                                    torch.ones((batch_size, 1)).cuda())
        Loss_domain_private_target = self.domain_loss(pred_domain_label_private_target,
                                                      torch.ones((batch_size, 1)).cuda())

        Loss_domain = Loss_domain_private_source + Loss_domain_private_target + Loss_domain_share_target + Loss_domain_share_source

        Loss_diff = self.diff_loss(feature_share_source, feature_private_source)
        Loss_diff += self.diff_loss(feature_share_target, feature_private_target)

        if self.Use_Spurious_Label:
            Loss = self.domain_loss_weight * Loss_domain + self.diff_loss_weight * Loss_diff + Loss_cls_target * self.target_loss_weight + Loss_cls

        else:
            Loss = self.domain_loss_weight * Loss_domain + self.diff_loss_weight * Loss_diff + Loss_cls

        self.log('Loss_domain_share_source', Loss_domain_share_source, prog_bar=True)
        self.log('Loss_domain_private_source', Loss_domain_private_source, prog_bar=True)
        self.log('Loss_domain_share_target', Loss_domain_share_target, prog_bar=True)
        self.log('Loss_domain_private_target', Loss_domain_private_target, prog_bar=True)

        self.log('Loss_cls', Loss_cls, prog_bar=True)
        if self.Use_Spurious_Label:
            self.log('Loss_cls_target', Loss_cls_target, prog_bar=True)
        self.log('Loss_domain', Loss_domain, prog_bar=True)
        self.log('Loss_diff', Loss_diff, prog_bar=True)
        self.log('Loss', Loss, prog_bar=True)
        if batch_idx % self.acc_check_step == 0:
            output = F.softmax(pred_class_label, dim=1)
            self.log('train_acc', accuracy(output, label), prog_bar=True)

        return Loss

    def validation_step(self, batch, batch_idx, dataloader_idx):
        x, y = batch
        output = self.network(x)
        output = F.softmax(output, dim=1)
        acc = accuracy(output, y)
        if dataloader_idx == 0:
            self.val_acc_source.append(acc)
        elif dataloader_idx == 1:
            self.val_acc_target.append(acc)

    def on_validation_epoch_end(self):
        final_acc_source = sum(self.val_acc_source) / len(self.val_acc_source)
        final_acc_target = sum(self.val_acc_target) / len(self.val_acc_target)
        self.val_acc_source.clear()
        self.val_acc_target.clear()

        self.log('val_final_acc_source', final_acc_source, prog_bar=True)
        self.log('val_final_acc_target', final_acc_target, prog_bar=True)

    def test_step(self, batch, batch_idx, dataloader_idx):
        x, y = batch
        output = self.network(x)
        output = F.softmax(output, dim=1)
        acc = accuracy(output, y)
        if dataloader_idx == 0:
            self.test_acc_source.append(acc)
        elif dataloader_idx == 1:
            self.test_acc_target.append(acc)

    def on_test_epoch_end(self):
        final_acc_source = sum(self.test_acc_source) / len(self.test_acc_source)
        final_acc_target = sum(self.test_acc_target) / len(self.test_acc_target)
        self.test_acc_source.clear()
        self.test_acc_target.clear()
        self.log('test_final_acc_source', final_acc_source, prog_bar=True)
        self.log('test_final_acc_target', final_acc_target, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        return optimizer
