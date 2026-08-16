import torch.nn as nn
import torch
# import resnet
import torch.nn.functional as F
from src.models.SegmentationTask.utils.block import *
from src.models.SegmentationTask.utils.ASPP import ASPPModule
from src.registries import NETWORK_REGISTRY
import hydra
bn_mom = 0.0003

class unet_encoder(torch.nn.Module):
    def __init__(self,num_band):
        super(unet_encoder, self).__init__() ##parent's init func
        self.conv1 = DoubleConv(num_band, 32, False)
        self.conv2 = torch.nn.Sequential(
            torch.nn.MaxPool2d(stride=2,kernel_size=2),
            DoubleConv(32, 64, False)
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.MaxPool2d(stride=2,kernel_size=2),
            DoubleConv(64, 128, False)
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.MaxPool2d(stride=2,kernel_size=2),
            DoubleConv(128, 256, False)
        )
        self.conv5 = torch.nn.Sequential(
            torch.nn.MaxPool2d(stride=2,kernel_size=2),
            DoubleConv(256, 512, False)
        )

    def forward(self,x):
        copies=[]#copies for upsample
        x=self.conv1(x)
        copies.append(x)    #X1
        x=self.conv2(x)
        copies.append(x)    #X2
        x=self.conv3(x)
        copies.append(x)    #X4
        x=self.conv4(x)
        copies.append(x)    #X8
        x=self.conv5(x)
        copies.append(x)    #X16
        # x=self.conv6(x)     #X16
        # copies.append(x)
        # x=self.conv7(x)
        return copies


class bridge(torch.nn.Module):  #X16 -> X32
    def __init__(self, in_chn, output_stride = 1):
        super(bridge, self).__init__() ##parent's init func
        if output_stride not in [1, 2]:
            raise ValueError('UNet.py: invalid output_stride')
        self.downsample = torch.nn.MaxPool2d(stride=2, kernel_size=2) if output_stride == 2 else None
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, in_chn * 2, kernel_size=3,stride = output_stride, padding=1),
            nn.BatchNorm2d(in_chn * 2, momentum=bn_mom),
            torch.nn.ReLU(),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn * 2, in_chn, kernel_size=1,stride = 1, padding=0),
            nn.BatchNorm2d(in_chn, momentum=bn_mom),
            torch.nn.ReLU(),
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            x = self.downsample(x)
        return out + x


class unet_decoder(torch.nn.Module):
    def __init__(self, num_class):
        super(unet_decoder,self).__init__() ##parent's init func
        # self.cat1=cat(2048,2048, 2048, upsample = False)
        self.cat2=cat(512,512, 512, upsample = True)
        self.cat3=cat(256,256, 256, upsample = True)
        self.cat4=cat(128,128, 128, upsample = True)
        self.cat5=cat(64,64, 64, upsample = True)
        self.cat6=cat(32,32,32, upsample = True)

        # self.conv1=torch.nn.Sequential(
        #     DoubleConv(2048,1024, False),
        #     nn.BatchNorm2d(1024, momentum=bn_mom),
        #     torch.nn.ReLU(),
        #     # torch.nn.Upsamle(scale_factor=2,mode='bilinear',align_corners=True)
        # )
        self.conv2 = DoubleConv(512, 256, False)
        self.conv3 = DoubleConv(256, 128, False)
        self.conv4 = DoubleConv(128, 64, False)
        self.conv5 = DoubleConv(64, 32, False)
        self.conv6 = DoubleConv(32, 16, False)
        self.conv7 = torch.nn.Conv2d(16,num_class,kernel_size=1,stride=1,padding=0)

    def forward(self,x,copies):
        # import ipdb
        # ipdb.set_trace()
        # x=self.cat1(x,copies[5])
        # x=self.conv1(x)
        x=self.cat2(x,copies[4])
        x=self.conv2(x)
        x=self.cat3(x,copies[3])
        x=self.conv3(x)
        x=self.cat4(x,copies[2])
        x=self.conv4(x)
        x=self.cat5(x,copies[1])
        x=self.conv5(x)
        x=self.cat6(x,copies[0])
        x=self.conv6(x)
        y=self.conv7(x)
        return y


class unet(torch.nn.Module):
    def __init__(self, num_band=3, num_class=1, mode='seg', **kwargs):
        super(unet,self).__init__() ##parent's init func
        self.num_class = num_class
        if mode == 'change':
            num_band = num_band*2
        self.encoder = unet_encoder(num_band)
        self.bridge = bridge(512, 2)
        self.decoder=unet_decoder(num_class)
    
    def forward(self,x):
        copies=self.encoder(x)
        x = self.bridge(copies[-1])
        y=self.decoder(x,copies)
        return y

@NETWORK_REGISTRY.register("res_unet")
class res_unet(torch.nn.Module):
    def __init__(self, backbone, num_class=1, **kwargs):
        super(res_unet,self).__init__() ##parent's init func
        self.backbone = backbone

        # 根据backbone类型设置channels_blocks
        if backbone.structure_type in ["resnet18", "resnet34"]:
            self.channels_blocks = [512, 256, 128, 64]
        else:  # resnet50, resnet101, resnet152, resnext等
            self.channels_blocks = [2048, 1024, 512, 256]
            
        # 根据output_stride设置do_upsample
        if backbone.outstride == 32:
            self.do_upsample = [True, True, True, True]
        elif backbone.outstride == 16:
            self.do_upsample = [False, True, True, True]
        else:  # outstride == 8
            self.do_upsample = [False, False, True, True]

        self.num_base_layers = len(self.channels_blocks)

        self.center = bridge(self.channels_blocks[0], 1)

        self.concat_blocks = nn.ModuleList([cat(self.channels_blocks[0],self.channels_blocks[0], self.channels_blocks[0], False)])
        self.decode_blocks = nn.ModuleList([DoubleConv(self.channels_blocks[0], self.channels_blocks[1], False)])
        for i in range(1, self.num_base_layers):
            self.concat_blocks.append(cat(self.channels_blocks[i],self.channels_blocks[i], self.channels_blocks[i], self.do_upsample[i]))
            if i < self.num_base_layers-1:
                self.decode_blocks.append(DoubleConv(self.channels_blocks[i], self.channels_blocks[i+1], False))
            else:
                self.decode_blocks.append(DoubleConv(self.channels_blocks[i], self.channels_blocks[i], False))

        self.segmentation_head = nn.Sequential(
                        nn.Conv2d(self.channels_blocks[-1], 16, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(16, momentum=bn_mom),
                        nn.ReLU(inplace=True),
                        torch.nn.Conv2d(16,num_class,kernel_size=1,stride=1,padding=0)
                        )
    
    def forward(self, x):
        x0_h, x0_w = x.size(2), x.size(3)
        layers = self.backbone(x)
        y = self.center(layers[-1])
        for i in range(self.num_base_layers) :
            y = self.concat_blocks[i](y, layers[self.num_base_layers-i-1])
            y = self.decode_blocks[i](y)

        y = F.interpolate(y, size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        y = self.segmentation_head(y)
        
        return y

@NETWORK_REGISTRY.register("SE_res_unet")
class SERes_UNet(torch.nn.Module):
    def __init__(self, backbone, num_class=1, **kwargs):
        super(SERes_UNet,self).__init__() ##parent's init func
        self.backbone = backbone

        # 根据backbone类型设置channels_blocks
        if backbone.structure_type in ["resnet18", "resnet34"]:
            self.channels_blocks = [512, 256, 128, 64]
        else:  # resnet50, resnet101, resnet152, resnext等
            self.channels_blocks = [2048, 1024, 512, 256]
            
        # 根据output_stride设置do_upsample
        if backbone.outstride == 32:
            self.do_upsample = [True, True, True, True]
        elif backbone.outstride == 16:
            self.do_upsample = [False, True, True, True]
        else:  # outstride == 8
            self.do_upsample = [False, False, True, True]

        self.num_base_layers = len(self.channels_blocks)

        self.center = bridge(self.channels_blocks[0], 1)

        self.concat_blocks = nn.ModuleList([cat(self.channels_blocks[0],self.channels_blocks[0], self.channels_blocks[0], False)])
        self.decode_blocks = nn.ModuleList([SeDoubleConv(self.channels_blocks[0], self.channels_blocks[1])])
        for i in range(1, self.num_base_layers):
            self.concat_blocks.append(cat(self.channels_blocks[i],self.channels_blocks[i], self.channels_blocks[i], self.do_upsample[i]))
            if i < self.num_base_layers-1:
                self.decode_blocks.append(SeDoubleConv(self.channels_blocks[i], self.channels_blocks[i+1]))
            else:
                self.decode_blocks.append(SeDoubleConv(self.channels_blocks[i], self.channels_blocks[i]))

        self.ASPP = ASPPModule(self.channels_blocks[-1], self.channels_blocks[-1])
        self.segmentation_head = nn.Sequential(
                        nn.Conv2d(self.channels_blocks[-1], 16,kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(16, momentum=bn_mom),
                        nn.ReLU(inplace=True),
                        torch.nn.Conv2d(16,num_class,kernel_size=1,stride=1,padding=0)
                        )
    
    def forward(self, x):
        x0_h, x0_w = x.size(2), x.size(3)
        layers = self.backbone(x)
        y = self.center(layers[-1])
        for i in range(self.num_base_layers) :
            y = self.concat_blocks[i](y, layers[self.num_base_layers-i-1])
            y = self.decode_blocks[i](y)
        y = self.ASPP(y)
        y = F.interpolate(y, size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        y = self.segmentation_head(y)
        
        return y

@NETWORK_REGISTRY.register("res_unet_aux")
class res_unet_aux(res_unet):
    def __init__(self, backbone, num_class=1, **kwargs):
        super(res_unet_aux,self).__init__(backbone, num_class, **kwargs) ##parent's init func
        
        self.segmentation_head_aux = nn.Sequential(
                        nn.Conv2d(self.channels_blocks[0], 16,kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(16, momentum=bn_mom),
                        nn.ReLU(inplace=True),
                        torch.nn.Conv2d(16,num_class,kernel_size=1,stride=1,padding=0)
                        )
    
    def forward(self, x):
        x0_h, x0_w = x.size(2), x.size(3)
        layers = self.backbone(x)
        y = self.center(layers[-1])
        y_aux = self.segmentation_head_aux(y)
        for i in range(self.num_base_layers) :
            y = self.concat_blocks[i](y, layers[self.num_base_layers-i-1])
            y = self.decode_blocks[i](y)

        y = F.interpolate(y, size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        y = self.segmentation_head(y)
        return [y, y_aux]

__all__ = [
    "res_unet",
    "res_unet_aux",
    "SERes_UNet",
]