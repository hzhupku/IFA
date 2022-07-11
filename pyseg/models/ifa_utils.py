import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats=128):
        super().__init__()
        self.row_embed = nn.Embedding(100, num_pos_feats)
        self.col_embed = nn.Embedding(100, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, x):
        # input: x, [b, N, 2]
        # output: [b, N, C]

        h = w = int(np.sqrt(x.shape[1]))
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1).view(x.shape[0],h*w, -1)
        #print('pos', pos.shape)
        return pos

class SpatialEncoding(nn.Module):
    def __init__(self, 
                 in_dim, 
                 out_dim, 
                 sigma = 6,
                 cat_input=True,
                 require_grad=False,):

        super().__init__()
        assert out_dim % (2*in_dim) == 0, "dimension must be dividable"

        n = out_dim // 2 // in_dim
        m = 2**np.linspace(0, sigma, n)
        m = np.stack([m] + [np.zeros_like(m)]*(in_dim-1), axis=-1)
        m = np.concatenate([np.roll(m, i, axis=-1) for i in range(in_dim)], axis=0)
        self.emb = torch.FloatTensor(m)
        if require_grad:
            self.emb = nn.Parameter(self.emb, requires_grad=True)    
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.sigma = sigma
        self.cat_input = cat_input
        self.require_grad = require_grad

    def forward(self, x):
       
        if not self.require_grad:
            self.emb = self.emb.to(x.device)
        y = torch.matmul(x, self.emb.T)
        if self.cat_input:
            return torch.cat([x, torch.sin(y), torch.cos(y)], dim=-1)
        else:
            return torch.cat([torch.sin(y), torch.cos(y)], dim=-1)

def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret

def ifa_feat(res, size, stride=1, local=False):
    bs, hh, ww = res.shape[0], res.shape[-2], res.shape[-1]
    h , w = size
    coords = (make_coord((h,w)).cuda().flip(-1) + 1) / 2
    #coords = (make_coord((h,w)).flip(-1) + 1) / 2
    coords = coords.unsqueeze(0).expand(bs, *coords.shape)
    coords = (coords*2-1).flip(-1)

    feat_coords = make_coord((hh,ww), flatten=False).cuda().permute(2, 0, 1) .unsqueeze(0).expand(res.shape[0], 2, *(hh,ww))
    #feat_coords = make_coord((hh,ww), flatten=False).permute(2, 0, 1) .unsqueeze(0).expand(res.shape[0], 2, *(hh,ww))

    if local:
        vx_list = [-1, 1]
        vy_list = [-1, 1]
        eps_shift = 1e-6
        rel_coord_list = []
        q_feat_list = []
        area_list = []
    else:
        vx_list, vy_list, eps_shift = [0], [0], 0
    rx = stride / h 
    ry = stride / w 
    
    for vx in vx_list:
        for vy in vy_list:
            coords_ = coords.clone()
            coords_[:,:,0] += vx * rx + eps_shift
            coords_[:,:,1] += vy * ry + eps_shift
            coords_.clamp_(-1+1e-6, 1-1e-6)
            q_feat = F.grid_sample(res, coords_.flip(-1).unsqueeze(1),mode='nearest',align_corners=False)[:,:,0,:].permute(0,2,1)
            q_coord = F.grid_sample(feat_coords, coords_.flip(-1).unsqueeze(1),mode='nearest',align_corners=False)[:,:,0,:].permute(0,2,1)
            rel_coord = coords - q_coord
            rel_coord[:,:,0] *= hh#res.shape[-2]
            rel_coord[:,:,1] *= ww#res.shape[-1]
            if local:
                rel_coord_list.append(rel_coord)
                q_feat_list.append(q_feat)
                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                area_list.append(area+1e-9)

    if not local:
        return rel_coord, q_feat
    else:
        return rel_coord_list, q_feat_list, area_list

