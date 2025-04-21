# Copyright (c) Facebook, Inc. and its affiliates.
# --------------------------------------------------------

# Modified by Jitesh Jain

import logging
from typing import Callable, Dict, List, Optional, Tuple, Union

import fvcore.nn.weight_init as weight_init
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.utils import _pair

from src.models.SegmentationTask.utils.detectron2_layers import Conv2d, ShapeSpec, get_norm, ModulatedDeformConv, _ModulatedDeformConv
# from detectron2.modeling import SEM_SEG_HEADS_REGISTRY

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

modulated_deform_conv = _ModulatedDeformConv.apply

class ModulatedDeformConvPack(ModulatedDeformConv):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 deformable_groups=1,
                 bias=True,
                 extra_offset_mask=False,):
        super(ModulatedDeformConvPack, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, deformable_groups, bias)
        self.extra_offset_mask = extra_offset_mask  #modified by lukx for fapn
        self.conv_offset_mask = nn.Conv2d(
            self.in_channels // self.groups,
            self.deformable_groups * 3 * self.kernel_size[0] *
            self.kernel_size[1],
            kernel_size=self.kernel_size,
            stride=_pair(self.stride),
            padding=_pair(self.padding),
            bias=True)
        self.init_offset()

    def init_offset(self):
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def forward(self, input):
        if self.extra_offset_mask:
            out = self.conv_offset_mask(input[1])
            input = input[0]
        else:
            out = self.conv_offset_mask(input)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return modulated_deform_conv(
            input, offset, mask, self.weight, self.bias, self.stride,
            self.padding, self.dilation, self.groups, self.deformable_groups)



# def build_pixel_decoder(cfg, input_shape):
#     """
#     Build a pixel decoder from `cfg.MODEL.MASK_FORMER.PIXEL_DECODER_NAME`.
#     """
#     name = cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME
#     model = SEM_SEG_HEADS_REGISTRY.get(name)(cfg, input_shape)
#     forward_features = getattr(model, "forward_features", None)
#     if not callable(forward_features):
#         raise ValueError(
#             "Only SEM_SEG_HEADS with forward_features method can be used as pixel decoder. "
#             f"Please implement forward_features for {name} to only return mask features."
#         )
#     return model


class FeatureSelectionModule(nn.Module):
    def __init__(self, in_chan, out_chan, norm="GN"):
        super(FeatureSelectionModule, self).__init__()
        self.conv_atten = Conv2d(in_chan, in_chan, kernel_size=1, bias=False, norm=get_norm(norm, in_chan))
        self.sigmoid = nn.Sigmoid()
        self.conv = Conv2d(in_chan, out_chan, kernel_size=1, bias=False, norm=get_norm('', out_chan))
        
        weight_init.c2_xavier_fill(self.conv_atten)
        weight_init.c2_xavier_fill(self.conv)

    def forward(self, x):
        atten = self.sigmoid(self.conv_atten(F.avg_pool2d(x, x.size()[2:])))
        feat = torch.mul(x, atten)
        x = x + feat
        feat = self.conv(self.conv(x))
        return feat

class FeatureAlign(nn.Module):      # Without FSM
    def __init__(self, in_nc=128, out_nc=128, norm=None):
        super(FeatureAlign, self).__init__()
        self.lateral_conv = FeatureSelectionModule(in_nc, out_nc, norm="")
        self.offset = Conv2d(out_nc * 2, out_nc, kernel_size=1, stride=1, padding=0, bias=False, norm=norm)
        self.dcpack_L2 = ModulatedDeformConvPack(out_nc, out_nc, 3, stride=1, padding=1, dilation=1, deformable_groups=8, extra_offset_mask=True)
        self.relu = nn.ReLU(inplace=True)
        weight_init.c2_xavier_fill(self.offset)

    def forward(self, feat_l, feat_s, main_path=None):
        HW = feat_l.size()[2:]
        if feat_l.size()[2:] != feat_s.size()[2:]:
            feat_up = F.interpolate(feat_s, HW, mode='bilinear', align_corners=False)
        else:
            feat_up = feat_s
        feat_arm = self.lateral_conv(feat_l)  # 0~1 * feats
        offset = self.offset(torch.cat([feat_arm, feat_up * 2], dim=1))  # concat for offset by compute the dif
        feat_align = self.relu(self.dcpack_L2([feat_up, offset]))  # [feat, offset]
        return feat_align + feat_arm


# @SEM_SEG_HEADS_REGISTRY.register()
class PixelFANDecoder(nn.Module):
    def __init__(self, feature_channels: list, *, conv_dim: int, mask_dim: int, norm: Optional[Union[str, Callable]] = None,):
        """
        NOTE: this interface is experimental.
        Args:
            feature_channels: channels of the input features
            conv_dims: number of output channels for the intermediate conv layers.
            mask_dim: number of output channels for the final conv layer.
            norm (str or callable): normalization for all conv layers
        """
        super().__init__()
        #input_shape['res5'].channels = 256
        # input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        
        # self.in_features = [k for k, v in input_shape]  # starting from "res2" to "res5"

        lateral_convs = []
        align_convs = []
        output_convs = []

        use_bias = norm == ""
        for idx, in_channels in enumerate(feature_channels):
            if idx == len(feature_channels) - 1:
                output_norm = get_norm(norm, conv_dim)
                output_conv = Conv2d(in_channels, conv_dim, kernel_size=3, stride=1, padding=1, bias=use_bias, norm=output_norm, activation=F.relu,)
                
                weight_init.c2_xavier_fill(output_conv)
                self.add_module("layer_{}".format(idx + 1), output_conv)
                
                lateral_convs.append(None)
                align_convs.append(None)
                output_convs.append(output_conv)
            else:
                output_norm = get_norm(norm, conv_dim)
                lateral_conv = Conv2d(in_channels, conv_dim, kernel_size=1, bias=use_bias, norm=get_norm(norm, conv_dim))
                align_con = FeatureAlign(conv_dim, conv_dim, norm=output_norm)
                output_conv = Conv2d(conv_dim, conv_dim, kernel_size=3, stride=1, padding=1, bias=use_bias, norm=output_norm, activation=F.relu,)
                
                weight_init.c2_xavier_fill(lateral_conv)
                weight_init.c2_xavier_fill(output_conv)
                
                self.add_module("adapter_{}".format(idx + 1), lateral_conv)
                self.add_module("align_{}".format(idx + 1), align_con)
                self.add_module("layer_{}".format(idx + 1), output_conv)
                
                lateral_convs.append(lateral_conv)
                align_convs.append(align_con)
                output_convs.append(output_conv)
        # Place convs into top-down order (from low to high resolution) to make the top-down computation in forward clearer.
        self.lateral_convs = lateral_convs[::-1]
        self.align_convs = align_convs[::-1]
        self.output_convs = output_convs[::-1]

        self.mask_dim = mask_dim
        self.mask_features = Conv2d(conv_dim, mask_dim, kernel_size=3, stride=1, padding=1,)
        weight_init.c2_xavier_fill(self.mask_features)
        
        self.maskformer_num_feature_levels = 3  # always use 3 scales

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        ret = {}
        ret["input_shape"] = {k: v for k, v in input_shape.items() if k in cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES}
        ret["conv_dim"] = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        ret["norm"] = cfg.MODEL.SEM_SEG_HEAD.NORM
        return ret

    def forward_features(self, features):
        multi_scale_features = []
        num_cur_levels = 0
        # Reverse feature maps into top-down order (from low to high resolution)
        for idx in range(len(features)):
            x = features[idx]
            lateral_conv = self.lateral_convs[idx]
            align_conv = self.align_convs[idx]
            output_conv = self.output_convs[idx]
            # import ipdb;ipdb.set_trace()
            if lateral_conv is None:
                y = output_conv(x)
            else:
                cur_fpn = lateral_conv(x)
                # Following FPN implementation, we use nearest upsampling here
                y = cur_fpn + F.interpolate(y, size=cur_fpn.shape[-2:], mode="nearest")
                y = output_conv(align_conv(cur_fpn, y))
            if num_cur_levels < self.maskformer_num_feature_levels:
                multi_scale_features.append(y)
                num_cur_levels += 1
        return multi_scale_features + [self.mask_features(y)]

    def forward(self, features, targets=None):
        # logger = logging.getLogger(__name__)
        # logger.warning("Calling forward() may cause unpredicted behavior of PixelDecoder module.")
        return self.forward_features(features)