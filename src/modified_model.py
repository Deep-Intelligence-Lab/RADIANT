import numpy as np
import torch

import scipy.io
import matplotlib.pyplot as plt
import sys
import timm
from einops import rearrange
from models import VisionTransformer

class SignalProjection(torch.nn.Module):
    def __init__(self, in_chans=1, embed_dim=768):
        super().__init__()
        self.proj = torch.nn.Conv2d(in_chans, embed_dim,kernel_size=[1,80], stride=[1,80])

    def forward(self, x):
        x = self.proj(x)
        x = rearrange(x, "b e p1 p2 -> b (p1 p2) e", p1=x.shape[2],p2=1, e=768)
        return x

class SignalTransformer(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.proj = SignalProjection()
        self.encoder = VisionTransformer(**config)
    
    def forward(self, x):
        x = self.proj(x)
        x = self.encoder(x)
        return x

