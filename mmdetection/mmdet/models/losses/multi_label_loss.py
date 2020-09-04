import torch
import torch.nn as nn

from ..registry import LOSSES
from .utils import weighted_loss
from .cross_entropy_loss import binary_cross_entropy

@weighted_loss
def multi_label_loss(pred,
                     target,
                     weight=None,
                     reduction='mean',
                     avg_factor=None
                     ):
    num_img = len(target)
    labels_unique=[label.unique() for label in target]
    classes = torch.zeros([num_img,81])
    for i in range(num_img):
        classes[i,labels_unique[i]] = 1
    classes = classes.to(pred.device)
    loss_multi_label = binary_cross_entropy(pred=pred,
                                            label=classes,
                                            weight=weight,
                                            reduction=reduction,
                                            avg_factor=avg_factor)
    return loss_multi_label


@LOSSES.register_module
class MultiLabelLoss(nn.Module):
    def __init__(self,
                 reduction='mean',
                 use_sigmoid=True,
                 loss_weight=1.0
                 ):
        super(MultiLabelLoss,self).__init__()
        self.reduction = reduction
        self.use_sigmoid=use_sigmoid
        if not self.use_sigmoid:
            raise NotImplementedError
        self.loss_weight = loss_weight


    def forward(self,
                cls_scores,
                gt_labels,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.use_sigmoid:
            loss_cls = self.loss_weight * multi_label_loss(pred=cls_scores,
                                                           target=gt_labels,
                                                           weight=weight,
                                                           reduction=reduction,
                                                           avg_factor=avg_factor)
        else:
            raise NotImplementedError
        return loss_cls
