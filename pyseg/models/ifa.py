import torch
import torch.nn as nn
from torch.nn import functional as F

from .ifa_utils import SpatialEncoding, ifa_feat, PositionEmbeddingLearned
from torch.nn.parameter import Parameter

def get_syncbn():
    #return nn.BatchNorm2d
    return nn.SyncBatchNorm


class ifa_simfpn(nn.Module):
    def __init__(self, ultra_pe=False, pos_dim=40, sync_bn=False, num_classes=19, local=False, unfold=False, stride=1, learn_pe=False, require_grad=False, num_layer=2):

        super(ifa_simfpn, self).__init__()
        self.pos_dim = pos_dim
        self.ultra_pe = ultra_pe
        self.local = local
        self.unfold = unfold
        self.stride = stride
        self.learn_pe = learn_pe
        norm_layer = get_syncbn() if sync_bn else nn.BatchNorm1d
        if learn_pe:
            self.pos1 = PositionEmbeddingLearned(self.pos_dim//2)
            self.pos2 = PositionEmbeddingLearned(self.pos_dim//2)
            self.pos3 = PositionEmbeddingLearned(self.pos_dim//2)
            self.pos4 = PositionEmbeddingLearned(self.pos_dim//2)
        if ultra_pe:
            self.pos1 = SpatialEncoding(2, self.pos_dim, require_grad=require_grad)
            self.pos2 = SpatialEncoding(2, self.pos_dim, require_grad=require_grad)
            self.pos3 = SpatialEncoding(2, self.pos_dim, require_grad=require_grad)
            self.pos4 = SpatialEncoding(2, self.pos_dim, require_grad=require_grad)
            self.pos_dim += 2
        
        in_dim = 4*(256 + self.pos_dim)

        if unfold:
            in_dim = 4*(256*9 + self.pos_dim)
     
       
        if num_layer == 2:
            self.imnet = nn.Sequential(
            nn.Conv1d(in_dim, 512, 1), norm_layer(512), nn.ReLU(), 
            nn.Conv1d(512, 256, 1), norm_layer(256), nn.ReLU(),
            nn.Conv1d(256, 256, 1), norm_layer(256), nn.ReLU(),
            nn.Conv1d(256, num_classes, 1)
            )
        elif num_layer == 0:
            self.imnet = nn.Sequential(
            nn.Conv1d(in_dim, 128, 1), norm_layer(128), nn.ReLU(), 
            nn.Conv1d(128, 128, 1), norm_layer(128), nn.ReLU(),
            nn.Conv1d(128, num_classes, 1)
            )               
        else:
            self.imnet = nn.Sequential(
            nn.Conv1d(in_dim, 512, 1), norm_layer(512), nn.ReLU(), 
            nn.Conv1d(512, 256, 1), norm_layer(256), nn.ReLU(),
            nn.Conv1d(256, num_classes, 1)
            )                  

    def forward(self, x, size, level=0, after_cat=False):
        h, w = size
        if not after_cat:
            if not self.local:
                if self.unfold:
                    x = F.unfold(x, 3, padding=1).view(x.shape[0],x.shape[1]*9, x.shape[2], x.shape[3])
                rel_coord, q_feat = ifa_feat(x, [h, w])
                if self.ultra_pe or self.learn_pe:
                    rel_coord = eval('self.pos'+str(level))(rel_coord)
                x = torch.cat([rel_coord, q_feat], dim=-1)
            else:
                if self.unfold:
                    x = F.unfold(x, 3, padding=1).view(x.shape[0],x.shape[1]*9, x.shape[2], x.shape[3])
                rel_coord_list, q_feat_list, area_list = ifa_feat(x, [h, w],  local=True, stride=self.stride)
                total_area = torch.stack(area_list).sum(dim=0)
                context_list = []
                for rel_coord, q_feat, area in zip(rel_coord_list, q_feat_list, area_list):
                    if self.ultra_pe or self.learn_pe:
                        rel_coord = eval('self.pos'+str(level))(rel_coord)
                    context_list.append(torch.cat([rel_coord, q_feat], dim=-1))
                ret = 0
                t = area_list[0]; area_list[0] = area_list[3]; area_list[3] = t
                t = area_list[1]; area_list[1] = area_list[2]; area_list[2] = t
                for conte, area in zip(context_list, area_list):
                    x = ret + conte *  ((area / total_area).unsqueeze(-1))              

        else:
            x = self.imnet(x).view(x.shape[0], -1, h, w)
        return x

        
