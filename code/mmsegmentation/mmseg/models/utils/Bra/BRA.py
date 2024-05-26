from collections import OrderedDict
from functools import partial
from typing import Tuple, Union

import torch
import torch.nn as nn

from timm.models.layers import DropPath, LayerNorm2d, to_2tuple, trunc_normal_

from .ops.bra_nchw import nchwBRA
from ._common import nchwAttentionLePE



class TBiSegBlock(nn.Module):

    """
    Attention + FFN
    """
    def __init__(self, dim, drop_path=0., num_heads=8, n_win=7, 
                       qk_scale=None, topk=4, mlp_ratio=4, side_dwconv=5, 
                       norm_layer=LayerNorm2d):

        super().__init__()
        self.norm1 = norm_layer(dim) # important to avoid attention collapsing
        
        if topk > 0:
            self.attn = nchwBRA(dim=dim, num_heads=num_heads, n_win=n_win,
                qk_scale=qk_scale, topk=topk, side_dwconv=side_dwconv)
        elif topk == -1:
            self.attn = nchwAttentionLePE(dim=dim)
        else:
            raise ValueError('topk should >0 or =-1 !')

        self.norm2 = norm_layer(dim)
        self.mlp = nn.Sequential(nn.Conv2d(dim, int(mlp_ratio*dim), kernel_size=1),
                                 nn.GELU(),
                                 nn.Conv2d(int(mlp_ratio*dim), dim, kernel_size=1)
                                )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
            

    def forward(self, x):
        """
        Args:
            x: NCHW tensor
        Return:
            NCHW tensor
        """
        # attention & mlp
        x = x + self.drop_path(self.attn(self.norm1(x))) # (N, C, H, W)
        x = x + self.drop_path(self.mlp(self.norm2(x))) # (N, C, H, W)
        return x

class BasicLayer(nn.Module):

    def __init__(self, dim, depth, num_heads, n_win, topk,
                 mlp_ratio=4., drop_path=0., side_dwconv=5):

        super().__init__()
        self.dim = dim
        self.depth = depth

        self.blocks = nn.ModuleList([
            TBiSegBlock(
                    dim=dim,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    num_heads=num_heads,
                    n_win=n_win,
                    topk=topk,
                    mlp_ratio=mlp_ratio,
                    side_dwconv=side_dwconv,
                )
            for i in range(depth)
        ])

    def forward(self, x:torch.Tensor):
        """
        Args:
            x: NCHW tensor
        Return:
            NCHW tensor
        """
        for blk in self.blocks:
            x = blk(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, depth={self.depth}"
    
