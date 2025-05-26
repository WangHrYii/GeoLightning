import torch
import torch.nn as nn
import sys

from torchvision import transforms
import torch.nn.functional as F
from einops import rearrange

import math
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image
import numpy as np
import torchvision.transforms as T

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        image2 = torch.clone(image)
        if len(image2.shape) == 4:
            # batched
            image2 = image2.permute(1, 0, 2, 3)
        for t, m, s in zip(image2, self.mean, self.std):
            t.mul_(s).add_(m)
        return image2.permute(1, 0, 2, 3)

norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
unnorm = UnNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

class ChannelNorm(torch.nn.Module):
    ### Not compatible with DDP

    def __init__(self, dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.norm = torch.nn.LayerNorm(dim)

    def forward(self, x):
        new_x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return new_x


class LayerNorm(nn.Module):
    """
    A LayerNorm variant, popularized by Transformers, that performs point-wise mean and
    variance normalization over the channel dimension for inputs that have shape
    (batch_size, channels, height, width).
    https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa B950
    """

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

# MinMaxScaler 缩放到[-0.5, 0.5]
class MinMaxScaler(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        c = x.shape[1]
        flat_x = x.permute(1, 0, 2, 3).reshape(c, -1)
        flat_x_min = flat_x.min(dim=-1).values.reshape(1, c, 1, 1)
        flat_x_scale = flat_x.max(dim=-1).values.reshape(1, c, 1, 1) - flat_x_min
        return ((x - flat_x_min) / flat_x_scale.clamp_min(0.0001)) - .5

class ImplicitFeaturizer(torch.nn.Module):

    def __init__(self, color_feats=True, n_freqs=10, learn_bias=False, time_feats=False, lr_feats=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.color_feats = color_feats
        self.time_feats = time_feats
        self.n_freqs = n_freqs
        self.learn_bias = learn_bias

        self.dim_multiplier = 2

        if self.color_feats:
            self.dim_multiplier += 3

        if self.time_feats:
            self.dim_multiplier += 1

        if self.learn_bias:
            self.biases = torch.nn.Parameter(torch.randn(2, self.dim_multiplier, n_freqs).to(torch.float32))
        
        self.low_res_feat = lr_feats

    def forward(self, original_image):
        b, c, h, w = original_image.shape
        grid_h = torch.linspace(-1, 1, h, device=original_image.device)
        grid_w = torch.linspace(-1, 1, w, device=original_image.device)
        feats = torch.cat([t.unsqueeze(0) for t in torch.meshgrid([grid_h, grid_w])]).unsqueeze(0)
        feats = torch.broadcast_to(feats, (b, feats.shape[1], h, w))

        if self.color_feats:
            feat_list = [feats, original_image]
        else:
            feat_list = [feats]

        feats = torch.cat(feat_list, dim=1).unsqueeze(1)
        freqs = torch.exp(torch.linspace(-2, 10, self.n_freqs, device=original_image.device)) \
            .reshape(1, self.n_freqs, 1, 1, 1) # torch.Size([1, 30, 1, 1, 1])
        feats = (feats * freqs) # torch.Size([1, 30, 5, 224, 224])

        if self.learn_bias:
            sin_feats = feats + self.biases[0].reshape(1, self.n_freqs, self.dim_multiplier, 1, 1) # torch.Size([1, 30, 5, 224, 224])
            cos_feats = feats + self.biases[1].reshape(1, self.n_freqs, self.dim_multiplier, 1, 1) # torch.Size([1, 30, 5, 224, 224])
        else:
            sin_feats = feats
            cos_feats = feats

        sin_feats = sin_feats.reshape(b, self.n_freqs * self.dim_multiplier, h, w) # torch.Size([1, 150, 224, 224])
        cos_feats = cos_feats.reshape(b, self.n_freqs * self.dim_multiplier, h, w) # torch.Size([1, 150, 224, 224])

        if self.color_feats:
            all_feats = [torch.sin(sin_feats), torch.cos(cos_feats), original_image]
        else:
            all_feats = [torch.sin(sin_feats), torch.cos(cos_feats)]

        if self.low_res_feat is not None:
            upsampled_feats = F.interpolate(self.low_res_feat, size=(h, w), mode='bilinear', align_corners=False)
            all_feats.append(upsampled_feats)

        return torch.cat(all_feats, dim=1)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class CrossAttentionLayer(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)  # Norm for query
        self.norm_kv = nn.LayerNorm(dim)  # Norm for key/value
        self.attention = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout)

    def forward(self, query, key, value):
        # Apply layer normalization
        query = self.norm_q(query)
        key = self.norm_kv(key)
        value = self.norm_kv(value)

        # Multi-head attention takes (sequence_length, batch_size, embedding_dim)
        query = query.permute(1, 0, 2)  # (seq_len, batch_size, dim)
        key = key.permute(1, 0, 2)      # (seq_len, batch_size, dim)
        value = value.permute(1, 0, 2)  # (seq_len, batch_size, dim)

        # Apply multi-head attention (cross-attention)
        attn_output, _ = self.attention(query, key, value)

        # Return to original format (batch_size, seq_len, dim)
        attn_output = attn_output.permute(1, 0, 2)
        return attn_output

class CATransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                CrossAttentionLayer(dim, heads=heads, dim_head=dim_head, dropout=dropout),  # Cross-Attention
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, query, key_value):
        for cross_attn, ff in self.layers:
            query = cross_attn(query, key_value, key_value) + query  # Cross-Attention
            # query = cross_attn(query, key_value, key_value) ## Because we are transforming imgs to features, we don't need to add the query back
            query = ff(query) + query  # Feed-Forward residual connection

        return self.norm(query)