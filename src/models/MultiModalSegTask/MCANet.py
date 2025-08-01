import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.MultiModalSegTask.MACAM import MCAM
from src.models.MultiModalSegTask.ASPP import ASPP
from src.models.MultiModalSegTask.ResNet import resnet101


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes, out_planes, stride=1, padding=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, bias=False)

class ConvBlock(nn.Module):
    """卷积块：Conv + BN + ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class UpBlock(nn.Module):
    """上采样块：上采样 + 跳跃连接 + 卷积细化"""
    def __init__(self, in_channels, skip_channels, out_channels, scale_factor=2):
        super(UpBlock, self).__init__()
        self.scale_factor = scale_factor
        self.up_conv = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        
        # 跳跃连接后的通道数
        total_channels = in_channels // 2 + skip_channels
        
        # 特征融合和细化
        self.conv_block = nn.Sequential(
            ConvBlock(total_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )
    
    def forward(self, x, skip=None):
        # 上采样
        x = self.up_conv(x)
        
        # 跳跃连接
        if skip is not None:
            # 确保尺寸匹配
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        
        # 特征细化
        x = self.conv_block(x)
        return x

class MACANet(nn.Module):
    def __init__(self, num_classes = 1000, pretrained = True, backbone = 'ResNet101', att_type=None):
        super(MACANet, self).__init__()
        # print(num_classes)
        self.encoder = EncoderBlock(pretrained, backbone, att_type=att_type)
        self.decoder = ImprovedDecoderBlock(num_classes)


    def forward(self, sar_img, opt_img):
        # 编码器现在返回多层特征和最终融合特征
        encoder_features = self.encoder.forward(sar_img, opt_img)
        classification = self.decoder(encoder_features)

        return classification

class EncoderBlock(nn.Module):
    def __init__(self, pretrained = True, backbone = 'ResNet101', num_classes=1000, att_type=None):
        super(EncoderBlock, self).__init__()
        if backbone == 'ResNet101':
            self.SAR_resnet = resnet101(pretrained, type='sar', num_classes=num_classes, att_type=att_type)
            self.OPT_resnet = resnet101(pretrained, type='opt', num_classes=num_classes, att_type=att_type)
        else:
            raise ValueError('Unsupported backbone - `{}`, Use ResNet101.'.format(backbone))

        self.MCAM_low = MCAM(in_channels=256)
        self.MCAM_mid = MCAM(in_channels=512)
        self.MCAM_high = MCAM(in_channels=2048)
        self.ASPP = ASPP(in_channels=2560, atrous_rates=[6, 12, 18])
        self.conv1 = conv1x1(2048, 256)
        self.conv2 = conv1x1(768, 48)
        self.conv3 = conv1x1(1024, 128)  # 用于中间层特征降维

    def forward(self, sar_img, opt_img):
        sar_feats = self.SAR_resnet.forward(sar_img)
        opt_feats = self.OPT_resnet.forward(opt_img)

        # 提取多层特征
        sar_low_feat = sar_feats[1]    # 256通道, 128x128
        sar_mid_feat = sar_feats[2]    # 512通道, 64x64  
        sar_high_feat = sar_feats[4]   # 2048通道, 16x16
        sar_final_feat = self.conv1(sar_feats[4])
        
        opt_low_feat = opt_feats[1]    # 256通道, 128x128
        opt_mid_feat = opt_feats[2]    # 512通道, 64x64
        opt_high_feat = opt_feats[4]   # 2048通道, 16x16
        opt_final_feat = self.conv1(opt_feats[4])

        # 多层特征融合
        low_level_features = self.MCAM_low(sar_low_feat, opt_low_feat)      # 256通道
        mid_level_features = self.MCAM_mid(sar_mid_feat, opt_mid_feat)      # 512通道
        high_level_features = self.MCAM_high(sar_high_feat, opt_high_feat)  # 2048通道

        # 构建跳跃连接特征
        # 128x128层
        low_level_sar_opt = torch.cat([sar_low_feat, opt_low_feat], 1)        # 512通道
        low_sar_opt_features = torch.cat([low_level_sar_opt, low_level_features], 1)  # 768通道
        skip_128 = self.conv2(low_sar_opt_features)  # 降维到48通道
        
        # 64x64层  
        mid_level_sar_opt = torch.cat([sar_mid_feat, opt_mid_feat], 1)        # 1024通道
        skip_64 = torch.cat([mid_level_sar_opt, mid_level_features], 1)       # 1536通道
        skip_64 = self.conv3(torch.cat([sar_mid_feat, opt_mid_feat], 1))      # 简化为128通道
        
        # 16x16层（ASPP处理后的高层特征）
        high_level_sar_opt = torch.cat([sar_final_feat, opt_final_feat], 1)   # 512通道
        high_sar_opt_features = torch.cat([high_level_sar_opt, high_level_features], 1)  # 2560通道
        high_features = self.ASPP(high_sar_opt_features)  # 256通道, 16x16

        return {
            'high_features': high_features,    # 256通道, 16x16
            'skip_64': skip_64,               # 128通道, 64x64  
            'skip_128': skip_128              # 48通道, 128x128
        }

class ImprovedDecoderBlock(nn.Module):
    def __init__(self, num_classes):
        super(ImprovedDecoderBlock, self).__init__()
        
        # 第一次上采样：16x16 -> 32x32
        self.up1 = UpBlock(256, 0, 128, scale_factor=2)  # 无跳跃连接
        
        # 第二次上采样：32x32 -> 64x64，融合64x64的跳跃连接
        self.up2 = UpBlock(128, 128, 64, scale_factor=2)
        
        # 第三次上采样：64x64 -> 128x128，融合128x128的跳跃连接
        self.up3 = UpBlock(64, 48, 32, scale_factor=2)
        
        # 第四次上采样：128x128 -> 256x256
        self.up4 = UpBlock(32, 0, 16, scale_factor=2)  # 无跳跃连接
        
        # 第五次上采样：256x256 -> 512x512
        self.up5 = UpBlock(16, 0, 16, scale_factor=2)  # 无跳跃连接
        
        # 最终分类层
        self.final_conv = nn.Conv2d(16, num_classes, kernel_size=1)
        
    def forward(self, encoder_features):
        x = encoder_features['high_features']  # 256通道, 16x16
        skip_64 = encoder_features['skip_64']   # 128通道, 64x64
        skip_128 = encoder_features['skip_128'] # 48通道, 128x128
        
        # 渐进式上采样
        x = self.up1(x)                    # 128通道, 32x32
        x = self.up2(x, skip_64)          # 64通道, 64x64，融合跳跃连接
        x = self.up3(x, skip_128)         # 32通道, 128x128，融合跳跃连接  
        x = self.up4(x)                   # 16通道, 256x256
        x = self.up5(x)                   # 16通道, 512x512
        
        # 最终分类
        output = self.final_conv(x)       # num_classes通道, 512x512
        
        return output

# 保持原有的DecoderBlock类以保证向后兼容
class DecoderBlock(nn.Module):
    def __init__(self, num_class):
        super(DecoderBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(256, num_class, kernel_size=1)
        )
    def forward(self, opt_sar_low_high_features):
        final_class = self.conv(opt_sar_low_high_features)
        final_img = F.interpolate(final_class, size=(512, 512), mode='bilinear', align_corners=False)

        return final_img