'''initialize'''
# from .builder import BackboneBuilder, BuildBackbone
# from .bricks import (
#     BuildDropout, BuildActivation, BuildNormalization, Scale, L2Norm, makedivisible, truncnormal, 
#     FFN, MultiheadAttention, nchwtonlc, nlctonchw, PatchEmbed, PatchMerging, AdaptivePadding, PositionEmbeddingSine,
#     DynamicConv2d, AdptivePaddingConv2d, SqueezeExcitationConv2d, DepthwiseSeparableConv2d, InvertedResidual, InvertedResidualV3,
#     DropoutBuilder, ActivationBuilder, NormalizationBuilder
# )

# 设置了project root后，可以直接import src中的模块
from src.models.backbones.builder import BackboneBuilder, BuildBackbone
from src.models.backbones.bricks import (
    BuildDropout, BuildActivation, BuildNormalization, Scale, L2Norm, makedivisible, truncnormal, 
    FFN, MultiheadAttention, nchwtonlc, nlctonchw, PatchEmbed, PatchMerging, AdaptivePadding, PositionEmbeddingSine,
    DynamicConv2d, AdptivePaddingConv2d, SqueezeExcitationConv2d, DepthwiseSeparableConv2d, InvertedResidual, InvertedResidualV3,
    DropoutBuilder, ActivationBuilder, NormalizationBuilder
)