'''
Function:
    Implementation of BackboneBuilder and BuildBackbone
Author:
    Zhenchao Jin
'''
import copy
# from .mae import MAE
# from .unet import UNet
# from .beit import BEiT
# from .cgnet import CGNet
# from .hrnet import HRNet
# from .erfnet import ERFNet
# from .resnet import ResNet
# from .samvit import SAMViT
# from .resnest import ResNeSt
# from .twins import PCPVT, SVT
# from .fastscnn import FastSCNN
# from .convnext import ConvNeXt
# from .bisenetv1 import BiSeNetV1
# from .bisenetv2 import BiSeNetV2
# from .swin import SwinTransformer
# from .convnextv2 import ConvNeXtV2
# from .vit import VisionTransformer
# from .mit import MixVisionTransformer
# from .timmwrapper import TIMMBackbone
# from .hiera import Hiera, HieraWithFPN
# from ...utils import BaseModuleBuilder
# from .edgesamrepvit import EdgeSAMRepViT
# from .mobilevit import MobileViT, MobileViTV2
# from .mobilesamtinyvit import MobileSAMTinyViT
# from .mobilenet import MobileNetV2, MobileNetV3
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
from src.utils import BaseModuleBuilder
from src.models.backbones.edgesamrepvit import EdgeSAMRepViT
from src.models.backbones.mobilevit import MobileViT, MobileViTV2
from src.models.backbones.mobilesamtinyvit import MobileSAMTinyViT
from src.models.backbones.mobilenet import MobileNetV2, MobileNetV3


'''BackboneBuilder'''
class BackboneBuilder(BaseModuleBuilder):
    REGISTERED_MODULES = {
        'UNet': UNet, 'BEiT': BEiT, 'CGNet': CGNet, 'HRNet': HRNet, 'MobileViT': MobileViT, 'MobileViTV2': MobileViTV2,
        'ERFNet': ERFNet, 'ResNet': ResNet, 'ResNeSt': ResNeSt, 'PCPVT': PCPVT, 'MobileSAMTinyViT': MobileSAMTinyViT, 
        'SVT': SVT, 'FastSCNN': FastSCNN, 'ConvNeXt': ConvNeXt, 'BiSeNetV1': BiSeNetV1, 'MAE': MAE, 'SAMViT': SAMViT,
        'SwinTransformer': SwinTransformer, 'VisionTransformer': VisionTransformer, 'EdgeSAMRepViT': EdgeSAMRepViT,
        'MixVisionTransformer': MixVisionTransformer, 'TIMMBackbone': TIMMBackbone, 'ConvNeXtV2': ConvNeXtV2, 'Hiera': Hiera,
        'MobileNetV2': MobileNetV2, 'MobileNetV3': MobileNetV3, 'BiSeNetV2': BiSeNetV2, 'HieraWithFPN': HieraWithFPN,
    }
    '''build'''
    def build(self, backbone_cfg):
        backbone_cfg = copy.deepcopy(backbone_cfg)
        if 'selected_indices' in backbone_cfg: backbone_cfg.pop('selected_indices')
        return super().build(backbone_cfg)


'''BuildBackbone'''
BuildBackbone = BackboneBuilder().build