import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils import ConvModule
from ..plugins import NonLocal2D
from mmdet.ops import DeformConv
class ImagePool(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(ImagePool,self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = ConvModule(in_channels,out_channels,1,1,0,1,conv_cfg=None,norm_cfg=None)

    def forward(self, x):
        _,_,H,W = x.shape
        h = self.pool(x)
        h = self.conv(h)
        h = F.interpolate(h,size=(H,W),mode='nearest')
        return h

class ASPP(nn.Module):
    def __init__(self,in_channels,out_channels,rates):
        super(ASPP, self).__init__()
        self.stages = nn.ModuleList()
        #non_local = NonLocal2D(
        #        in_channels,
        #        reduction=1,
        #        use_scale=False,
        #        conv_cfg=None,
        #        norm_cfg=None)
        #self.stages.append(non_local)
        self.offsets = nn.ModuleList()
        for i, rate in enumerate(rates):
            deform_conv_op = DeformConv(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=rate,
                dilation=rate,
                deformable_groups=1,
                bias=False)
            conv_offset = ConvModule(
                in_channels=in_channels,
                out_channels=18,
                kernel_size=3,
                stride=1,
                padding=rate,
                dilation=rate,
                conv_cfg=None,
                norm_cfg=None
            )
            self.stages.append(deform_conv_op)
            self.offsets.append(conv_offset)

    def forward(self,x):
        #return torch.cat([stage(x) for stage in self.stages.children()],dim=1)
        out = []
        #out.append(self.stages[0](x))
        for i in range(len(self.stages)):
            offset = self.offsets[i - 1](x)
            out.append(self.stages[i](x, offset))
        return torch.cat(out, dim=1)

class SENet(nn.Module):
    def __init__(self,in_channels):
        super(SENet,self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = ConvModule(in_channels,in_channels // 4,1,1,0,1,conv_cfg=None,norm_cfg=None,activation='relu')
        self.conv2 = ConvModule(in_channels // 4,in_channels,1,1,0,1,conv_cfg=None,norm_cfg=None,activation=None)


    def forward(self, x):
        _,_,H,W = x.shape
        h = self.pool(x)
        h = self.conv1(h)
        h = self.conv2(h)
        out = torch.sigmoid(h)
        return out

class ContextLearning(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(ContextLearning,self).__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.rates = [2,3,5,9]

        self.aspp = ASPP(in_channels=self.in_channels,out_channels=self.out_channels,rates=self.rates)
        self.senet = SENet(in_channels=self.in_channels*4)
        #self.non_nocal = NonLocal2D(
        #    in_channels,
        #    reduction=1,
        #    use_scale=False,
        #    conv_cfg=None,
        #    norm_cfg=None)

    def forward(self, inputs):
        b,c,h,w = inputs.shape
        inputs_aspp = self.aspp(inputs)
        aspp_senet = (1+self.senet(inputs_aspp)) * inputs_aspp
        aspp_senet = aspp_senet.permute(0,2,3,1).view([b,h,w,4,c]).sum(3)    #把aspp_senet 4层输出按像素对应相加，所以把1024拆分成4*256
        aspp_senet = aspp_senet.permute(0,3,1,2)
        #aspp_senet_nonlocal = self.non_nocal(aspp_senet)
        out = inputs+aspp_senet
        return out



