import torch
import torch.nn as nn

from model_lib.Classifier import DP_Classifier
from model_lib.Conv1D_Encoder import Encoder1D_Param
from model_lib.Domain_Discriminator import DomainDiscriminator
from model_lib.Feature_Encoder import DP_Feature_Encoder
from model_lib.RevGradLayer import ReverseLayerF


class DP_Net_CrossDomain(nn.Module):
    def __init__(self, classes_number, input_size, Encoder_Param, drop_rate=0.1,
                 original_compatible="non-conservative", hidden_size=128, discriminator_hidden_size=64,
                 target_domain_num=1):
        super().__init__()
        self.Feature_Encoder_Share = DP_Feature_Encoder(input_size, Encoder_Param, drop_rate, original_compatible,
                                                        hidden_size)
        self.Feature_Encoder_Source = DP_Feature_Encoder(input_size, Encoder_Param, drop_rate, original_compatible,
                                                         hidden_size)
        self.Feature_Encoder_Target = DP_Feature_Encoder(input_size, Encoder_Param, drop_rate, original_compatible,
                                                         hidden_size)

        self.Classifier = DP_Classifier(classes_number, hidden_size)

        self.Domain_Discriminator = DomainDiscriminator(hidden_size, discriminator_hidden_size, target_domain_num)

    def forward(self, x):
        if self.training:
            source, target, alpha = x
            feature_share_target = self.Feature_Encoder_Share(target)
            feature_share_source = self.Feature_Encoder_Share(source)
            feature_private_target = self.Feature_Encoder_Target(target)
            feature_private_source = self.Feature_Encoder_Source(source)

            feature_share_source_reverse = ReverseLayerF.apply(feature_share_source, alpha)
            feature_share_target_reverse = ReverseLayerF.apply(feature_share_target, alpha)

            pred_class_label = self.Classifier(feature_share_source)

            pred_domain_label_share_source = self.Domain_Discriminator(feature_share_source_reverse)
            pred_domain_label_share_target = self.Domain_Discriminator(feature_share_target_reverse)

            pred_domain_label_private_source = self.Domain_Discriminator(
                torch.add(feature_private_source, feature_share_source))
            pred_domain_label_private_target = self.Domain_Discriminator(
                torch.add(feature_private_target, feature_share_target))
            return (
                feature_share_target, feature_share_source, feature_private_target, feature_private_source,
                pred_class_label,
                pred_domain_label_share_source, pred_domain_label_share_target, pred_domain_label_private_source,
                pred_domain_label_private_target)
        else:
            y = self.Feature_Encoder_Share(x)
            y = self.Classifier(y)
            return y


if __name__ == "__main__":
    inputs1 = torch.randn(16, 1, 10000)
    inputs2 = torch.randn(16, 1, 10000)
    sk = DP_Net_CrossDomain(4, 1, Encoder1D_Param)
    out = sk(inputs1, inputs2, 0.3)
    criterion = nn.L1Loss()
    loss = criterion(out, inputs1)
    loss.backward()
    print(1)
