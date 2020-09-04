import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init

from ..plugins import NonLocal2D,ContextLearning
from ..registry import NECKS
from ..utils import ConvModule


@NECKS.register_module
class BFP(nn.Module):
    """BFP (Balanced Feature Pyrmamids)

    BFP takes multi-level features as inputs and gather them into a single one,
    then refine the gathered feature and scatter the refined results to
    multi-level features. This module is used in Libra R-CNN (CVPR 2019), see
    https://arxiv.org/pdf/1904.02701.pdf for details.

    Args:
        in_channels (int): Number of input channels (feature maps of all levels
            should have the same channels).
        num_levels (int): Number of input feature levels.
        conv_cfg (dict): The config dict for convolution layers.
        norm_cfg (dict): The config dict for normalization layers.
        refine_level (int): Index of integration and refine level of BSF in
            multi-level features from bottom to top.
        refine_type (str): Type of the refine op, currently support
            [None, 'conv', 'non_local'].
    """

    def __init__(self,
                 in_channels,
                 num_levels,
                 refine_level=2,
                 refine_type=None,
                 conv_cfg=None,
                 norm_cfg=None):
        super(BFP, self).__init__()
        assert refine_type in [None, 'conv', 'non_local','context_learning']

        self.in_channels = in_channels
        self.num_levels = num_levels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.refine_level = refine_level
        self.refine_type = refine_type


        assert 0 <= self.refine_level < self.num_levels

        if self.refine_type == 'conv':
            self.refine = ConvModule(
                self.in_channels,
                self.in_channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg)
        elif self.refine_type == 'non_local':
            self.refine = NonLocal2D(
                self.in_channels,
                reduction=1,
                use_scale=False,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg)
        elif self.refine_type == 'context_learning':
            self.refine = ContextLearning(
                self.in_channels,
                self.in_channels)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        #self.spatial_attention = nn.ModuleList()
        #for i in range(self.num_levels):
        #    sa = SpatialAttention(self.in_channels)
        #    self.spatial_attention.append(sa)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def context_refine(self,inputs,refine_type=None,
                                   refine_from_num_levels=[0,5],
                                   refine_for_num_levels=[0,5],
                                   refine_level = 2
                       ):
        """

        :param inputs: 输入的Tensor元组
        :param refine_type: 加强类型，如果为None，则是融合输入特征，分辨率增强
        :param refine_from_num_levels: 融合的特征层数,默认所有层
        :param refine_for_num_levels: 需要加强的层数,默认所有层（例如可以只融合中间几层特征，然后用来对每一层加强）
        :param refine_level: 其它层的尺寸调整至该层
        :return:
        """
        # step 1: gather multi-level features by resize and average
        feats = []
        gather_size = inputs[refine_level].size()[2:]
        for i in range(refine_from_num_levels[0],refine_from_num_levels[1]):
            if i < refine_level:
                gathered = F.adaptive_max_pool2d(
                    inputs[i], output_size=gather_size)
            else:
                gathered = F.interpolate(
                    inputs[i], size=gather_size, mode='nearest')
            feats.append(gathered)

        bsf = sum(feats) / len(feats)

        # step 2: refine gathered features
        if refine_type is not None:
            bsf = self.refine(bsf)

        # step 3: scatter refined features to multi-levels by a residual path
        outs = []
        for i in range(refine_for_num_levels[0],refine_for_num_levels[1]):
            out_size = inputs[i].size()[2:]
            if i < self.refine_level:
                residual = F.interpolate(bsf, size=out_size, mode='nearest')
            else:
                residual = F.adaptive_max_pool2d(bsf, output_size=out_size)

            #residual = self.spatial_attention[i](residual)*residual
            outs.append(residual)

        return outs


    def forward(self, inputs):
        #fpns,resnets = inputs[0],inputs[1]

        assert len(inputs) == self.num_levels

        context_refine = self.context_refine(inputs,
                                             refine_type='context_learning',
                                             refine_from_num_levels=[0,5],
                                             refine_for_num_levels=[0,5],
                                             refine_level=2
                                             )
        #resolution_refine = self.context_refine(resnets,
        #                                     refine_type= 'context_learning',
        #                                     refine_from_num_levels=[0, 5],
        #                                     refine_for_num_levels=[0, 5],
        #                                     refine_level=2
        #                                     )

        outs = [inputs[i]+context_refine[i] for i in range(self.num_levels)]
        return tuple(outs)