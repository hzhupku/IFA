import torch
import torch.nn as nn
from torch.nn import functional as F
from .base import PPM, ASPP, get_syncbn


class dec_deeplabv3(nn.Module):
    """
    Reference:
        Chen, Liang-Chieh, et al. *"Rethinking Atrous Convolution for Semantic Image Segmentation."*
    """
    def __init__(self, in_planes, num_classes=19, inner_planes=256, sync_bn=False, dilations=(12, 24, 36)):
        super(dec_deeplabv3, self).__init__()

        norm_layer = get_syncbn() if sync_bn else nn.BatchNorm2d

        self.aspp = ASPP(in_planes, inner_planes=inner_planes, sync_bn=sync_bn, dilations=dilations)
        self.head = nn.Sequential(
            nn.Conv2d(self.aspp.get_outplanes(), 256, kernel_size=3, padding=1, dilation=1, bias=False),
            norm_layer(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0, bias=True))

    def forward(self, x):
        #print('debug1',x.shape)
        aspp_out = self.aspp(x)
        res = self.head(aspp_out)
        return res


class dec_deeplabv3_plus(nn.Module):
    """
    Reference:
        Chen, Liang-Chieh, et al. *"Rethinking Atrous Convolution for Semantic Image Segmentation."*
    """
    def __init__(self, in_planes, num_classes=19, inner_planes=256, sync_bn=False, dilations=(12, 24, 36), ks=3):
        super(dec_deeplabv3_plus, self).__init__()

        norm_layer = get_syncbn() if sync_bn else nn.BatchNorm2d

        self.aspp = ASPP(in_planes, inner_planes=inner_planes, sync_bn=sync_bn, dilations=dilations)
        self.head = nn.Sequential(
            nn.Conv2d(self.aspp.get_outplanes(), 256, kernel_size=3, padding=1, dilation=1, bias=False),
            norm_layer(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1))
        self.final =  nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        self.tail = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=ks, stride=1, padding=1, bias=True),
            norm_layer(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, 256, kernel_size=ks, stride=1, padding=1, bias=True),
            norm_layer(256),nn.ReLU(inplace=True),nn.Dropout2d(0.1))
        self.low_conv = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1),
            norm_layer(256),
            nn.ReLU(inplace=True))
    def forward(self, x):
        x1, x2, x3, x4 = x
        #print('shape',x1.shape,x2.shape,x3.shape,x4.shape)
        aspp_out = self.aspp(x4)
        low_feat = self.low_conv(x1)
        aspp_out = self.head(aspp_out)
        h,w = low_feat.size()[-2:]
        aspp_out = F.interpolate(aspp_out,size=(h,w),mode='bilinear',align_corners=True)
        aspp_out = torch.cat((low_feat,aspp_out),dim=1)
        aspp_out = self.tail(aspp_out)
        res = self.final(aspp_out)
        return res


class Aux_Module(nn.Module):
    def __init__(self, in_planes, num_classes=19, sync_bn=False):
        super(Aux_Module, self).__init__()

        norm_layer = get_syncbn() if sync_bn else nn.BatchNorm2d
        self.aux = nn.Sequential(
                nn.Conv2d(in_planes, 256, kernel_size=3, stride=1, padding=1),#512
                norm_layer(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1),
                nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0, bias=True))

    def forward(self, x):
        res = self.aux(x)
        return res



