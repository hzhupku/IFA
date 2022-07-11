import torch
import torch.nn as nn
from torch.nn import functional as F
from .base import ASPP
from .ifa import ifa_simfpn
import math
import numpy as np

def get_syncbn():
    #return nn.BatchNorm2d
    return nn.SyncBatchNorm


class fpn_ifa(nn.Module):

    def __init__(self, in_planes, num_classes=19, inner_planes=256, sync_bn=False, dilations=(12, 24, 36), 
                pos_dim=24, ultra_pe=False, unfold=False, no_aspp=False,
                local=False, stride=1, learn_pe=False, require_grad=False, num_layer=2):

        super(fpn_ifa, self).__init__()
        norm_layer = get_syncbn() if sync_bn else nn.BatchNorm2d

        self.no_aspp = no_aspp

        self.unfold = unfold
      
        
        if self.no_aspp:
            self.head = nn.Sequential(nn.Conv2d(2048, 256, kernel_size=1), norm_layer(256), nn.ReLU(inplace=True))
        else:
            self.aspp = ASPP(in_planes, inner_planes=inner_planes, sync_bn=sync_bn, dilations=dilations)
            self.head = nn.Sequential(
                nn.Conv2d(self.aspp.get_outplanes(), 256, kernel_size=3, padding=1, dilation=1, bias=False),
                norm_layer(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1))   
        
        self.ifa = ifa_simfpn(ultra_pe=ultra_pe, pos_dim=pos_dim, sync_bn=sync_bn, num_classes=num_classes, local=local, unfold=unfold, stride=stride, learn_pe=learn_pe, require_grad=require_grad, num_layer=num_layer)
        self.enc1 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1), norm_layer(256), nn.ReLU(inplace=True))
        self.enc2 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=1), norm_layer(256), nn.ReLU(inplace=True))
        self.enc3 = nn.Sequential(nn.Conv2d(1024, 256, kernel_size=1), norm_layer(256), nn.ReLU(inplace=True))
    
    def forward(self, x):
        x1, x2, x3, x4 = x
        if self.no_aspp:
            aspp_out = self.head(x4)
        else:
            aspp_out = self.aspp(x4)
            aspp_out = self.head(aspp_out)

        x1 = self.enc1(x1)
        x2 = self.enc2(x2)
        x3 = self.enc3(x3)
        context = []
        h, w = x1.shape[-2], x1.shape[-1]


        target_feat = [x1, x2, x3, aspp_out]

        for i, feat in enumerate(target_feat):
            context.append(self.ifa(feat, size=[h, w], level=i+1))
        context = torch.cat(context, dim=-1).permute(0,2,1)

        res = self.ifa(context, size=[h, w], after_cat=True)

        return res
