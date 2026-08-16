'''initialize'''
# from .builder import BackboneBuilder, BuildBackbone
# from .bricks import (
#     BuildDropout, BuildActivation, BuildNormalization, Scale, L2Norm, makedivisible, truncnormal, 
#     FFN, MultiheadAttention, nchwtonlc, nlctonchw, PatchEmbed, PatchMerging, AdaptivePadding, PositionEmbeddingSine,
#     DynamicConv2d, AdptivePaddingConv2d, SqueezeExcitationConv2d, DepthwiseSeparableConv2d, InvertedResidual, InvertedResidualV3,
#     DropoutBuilder, ActivationBuilder, NormalizationBuilder
# )

# 设置了project root后，直接从import src中的模块

# 由于面向Hydra的配置设计，所以这里直接暴露写好的对象类，不需要注册机制，内部存在的一些注册机制是为了支持一些特殊的功能
from src.models.backbones.mae import MAE
from src.models.backbones.unet import UNet
from src.models.backbones.beit import BEiT
from src.models.backbones.cgnet import CGNet
from src.models.backbones.hrnet import HRNet
from src.models.backbones.erfnet import ERFNet
from src.models.backbones.resnet import ResNet
from src.models.backbones.samvit import SAMViT
from src.models.backbones.resnest import ResNeSt
from src.models.backbones.twins import PCPVT, SVT
from src.models.backbones.fastscnn import FastSCNN
from src.models.backbones.convnext import ConvNeXt
from src.models.backbones.bisenetv1 import BiSeNetV1
from src.models.backbones.bisenetv2 import BiSeNetV2
from src.models.backbones.swin import SwinTransformer
from src.models.backbones.convnextv2 import ConvNeXtV2
from src.models.backbones.vit import VisionTransformer
from src.models.backbones.mit import MixVisionTransformer
from src.models.backbones.timmwrapper import TIMMBackbone
from src.models.backbones.hiera import Hiera, HieraWithFPN
from src.models.backbones.edgesamrepvit import EdgeSAMRepViT
from src.models.backbones.mobilevit import MobileViT, MobileViTV2
from src.models.backbones.mobilesamtinyvit import MobileSAMTinyViT
from src.models.backbones.mobilenet import MobileNetV2, MobileNetV3
from src.models.backbones.dinov2 import DINOv2
from src.models.backbones.torchvision_source import (
    TorchvisionSourceBackbone,
    create_source_model,
    list_source_models,
)
from src.models.backbones.protocol import (
    FeatureBackbone,
    FeatureBackboneAdapter,
    adapt_backbone,
)

from src.models.backbones.bricks import (
    Scale, L2Norm, makedivisible, truncnormal, 
    FFN, MultiheadAttention, nchwtonlc, nlctonchw, PatchEmbed, PatchMerging, AdaptivePadding, PositionEmbeddingSine,
    DynamicConv2d, AdptivePaddingConv2d, SqueezeExcitationConv2d, DepthwiseSeparableConv2d, InvertedResidual, InvertedResidualV3,
    DropoutBuilder, ActivationBuilder, NormalizationBuilder
)

from .loft import LoftUp, apply_mask_optimization, load_loftup_checkpoint, get_upsampler

__all__ = [
    'LoftUp',
    'apply_mask_optimization',
    'load_loftup_checkpoint',
    'get_upsampler',
    'TorchvisionSourceBackbone',
    'create_source_model',
    'list_source_models',
    'FeatureBackbone',
    'FeatureBackboneAdapter',
    'adapt_backbone',
]
