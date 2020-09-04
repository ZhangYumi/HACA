import torch.nn as nn
from ..builder import build_loss
from ..registry import MULTI_LABEL
from mmdet.core import force_fp32

@MULTI_LABEL.register_module
class MultiLabel(nn.Module):
    def __init__(self,
                 in_channels,
                 num_classes,
                 loss_cls
                 ):
        super(MultiLabel,self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.global_pooling = nn.AdaptiveAvgPool2d([1,1])
        self.loss_cls = build_loss(loss_cls)
        self.fc_cls_1 = nn.Linear(self.in_channels,1024)
        self.fc_cls_2 = nn.Linear(1024, self.num_classes)
        self.relu = nn.ReLU(inplace=True)

    def init_weights(self):
        nn.init.normal_(self.fc_cls_1.weight, 0, 0.01)
        nn.init.constant_(self.fc_cls_1.bias, 0)

        nn.init.normal_(self.fc_cls_2.weight, 0, 0.01)
        nn.init.constant_(self.fc_cls_2.bias, 0)

    @force_fp32(apply_to=('cls_scores'))
    def loss(self,cls_scores,gt_labels):
        losses = dict()
        losses['multi_label_loss'] = self.loss_cls(cls_scores,gt_labels)
        return losses

    def forward(self, x):
        x = self.global_pooling(x)
        x = x.view(x.size()[0],-1)
        x = self.relu(self.fc_cls_1(x))
        x = self.fc_cls_2(x)
        return x
