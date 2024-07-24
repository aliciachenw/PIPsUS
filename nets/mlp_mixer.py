import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce


def MLPMixer(S, latent_dim, dim, depth, corr_levels, corr_radius, expansion_factor=4, dropout=0.):

    kitchen_dim = (corr_levels * (2*corr_radius + 1)**2) + latent_dim + 2
    
    chan_first, chan_last = partial(nn.Conv1d, kernel_size=1), nn.Linear

    return nn.Sequential(
        nn.Linear(kitchen_dim, dim),
        *[nn.Sequential(
            PreNormResidual(dim, FeedForward(S, expansion_factor, dropout, chan_first)),
            PreNormResidual(dim, FeedForward(dim, expansion_factor, dropout, chan_last))
        ) for _ in range(depth)],
        nn.LayerNorm(dim),
        Reduce('b n c -> b c', 'mean'),
        nn.Linear(dim, 2)
    )


class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x

def FeedForward(dim, expansion_factor=4, dropout=0., dense=nn.Linear):
    return nn.Sequential(
        dense(dim, dim * expansion_factor),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(dim * expansion_factor, dim),
        nn.Dropout(dropout)
    )