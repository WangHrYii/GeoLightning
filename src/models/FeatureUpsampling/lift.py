"""
LiFT Module for ViT feature upsampling.

Code by: Saksham Suri and Matthew Walmer
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.FeatureUpsampling.BaseModel import FeatureUpsamplingBase


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.conv_1 = DoubleConv(in_channels//2+32, out_channels//2)

    def forward(self, x, imgs_1):
        x = self.up(x)
        x = torch.cat([x, imgs_1], dim=1)
        x = self.conv_1(x)
        return x


"""
in_channels: number of channels in the features from the ViT backbone

patch_size: size of patches used by the ViT backbone

pre_shape: enable/disable reshaping of feature inputs, altering the expected input shape 
    True - will accept input shape [B, T, C] (ViT standard), which it will convert to [B, C, H, W]
    False - will accept input shape [B, C, H, W], conversion already performed

post_shape: enable/disable reshaping of feature outputs
    True - will return output in shape [B, T, C] (ViT standard)
    False - will return output in shape [B, C, H, W]
"""
class LiFT(nn.Module):
    def __init__(self, in_channels, patch_size, pre_shape=False, post_shape=False):
        super(LiFT, self).__init__()
        self.patch_size = patch_size
        self.pre_shape = pre_shape
        self.post_shape = post_shape
        
        self.up1 = (Up(in_channels+32, in_channels))
        self.outc = nn.Conv2d(in_channels//2, in_channels, kernel_size=1)
        self.image_convs_1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        if patch_size == 8:
            self.scale_adapter = nn.Identity()
        elif patch_size == 16:
            self.scale_adapter = nn.MaxPool2d(2, 2)
        elif patch_size == 14:
            self.scale_adapter = None
            # self.scale_adapter_14 = F.adaptive_max_pool2d((14, 14))
            # self.scale_adapter_7 = F.adaptive_max_pool2d((7, 7))
            # self.scale_adapter_28 = F.adaptive_max_pool2d((28, 28))
        else:
            print('ERROR: patch size %i not currently supported'%patch_size)
            exit()
        self.image_convs_2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )


    # [B, T, C] --> [B, C, H, W]
    def run_pre_shape(self, imgs, x):
        H = int(imgs.shape[2] / self.patch_size)
        W = int(imgs.shape[3] / self.patch_size)
        x = x.permute(0, 2, 1)
        x = x.reshape(x.shape[0], -1, H, W)
        return x


    # [B, C, H, W] --> [B, T, C]
    def run_post_shape(self, x):
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        return x


    def forward(self, x, imgs):
        if self.pre_shape: x = self.run_pre_shape(imgs, x)
        imgs_1 = self.image_convs_1(imgs)
        imgs_1 = F.adaptive_max_pool2d(imgs_1, (x.shape[2]*2, x.shape[3]*2))
        # imgs_1 = self.scale_adapter(imgs_1)
        imgs_2 = self.image_convs_2(imgs_1)
        # Enable the following if working with both --imsize 56 and --patch_size 16
        # if(x.shape[2] != imgs_2.shape[2]):
        #     imgs_1 = self.image_convs_1(imgs[:,:,2:-2,2:-2])
        #     imgs_1 = self.scale_adapter(imgs_1)
        #     imgs_2 = self.image_convs_2(imgs_1)
        x = torch.cat([x, imgs_2], dim=1)
        x = self.up1(x, imgs_1)
        logits = self.outc(x)
        if self.post_shape: logits = self.run_post_shape(logits)
        return logits

def load_lift_checkpoints(lift_path, channel=384, patch=14):
        lift = LiFT(channel, patch)
        state_dict = torch.load(lift_path)
            # if "module." in state_dict, remove it
        for k in list(state_dict.keys()):
            if k.startswith('module.'):
                state_dict[k[7:]] = state_dict[k]
                del state_dict[k]
        lift.load_state_dict(state_dict)
        lift.to("cuda")
        print('Loaded LiFT module from: ' + lift_path)
        return lift


class LiFTTrainer(FeatureUpsamplingBase):
    """
    LiFT模型训练器
    """
    def __init__(self, 
                 vfm_model,
                 lift_model,
                 optimizer=None,
                 alpha: float = 0.8,
                 vfm_input_size: int = 224,
                 patch_size: int = 14):
        """
        初始化LiFT训练器
        
        参数:
            vfm_model: 基础视觉模型，如DINOv2
            lift_model: LiFT上采样模型
            optimizer: 优化器
            alpha: 特征混合系数
            vfm_input_size: VFM输入尺寸(默认224)
            patch_size: 视觉模型的patch大小(默认14)
        """
        super().__init__(optimizer=optimizer, alpha=alpha)
        self.save_hyperparameters(ignore=['vfm_model', 'lift_model'])
        
        # 模型组件
        self.vfm_model = vfm_model
        self.lift_model = lift_model

        # 参数
        self.vfm_input_size = vfm_input_size
        self.patch_size = patch_size

        # 冻结基础模型
        for param in self.vfm_model.parameters():
            param.requires_grad = False
            
    def forward(self, x, original_img=None, img_path=None):
        """
        前向传播函数
        
        参数:
            x: 输入图像 [B, C, H, W]，已调整为VFM尺寸(224x224)
            original_img: 原始高分辨率图像，在此模型中不使用
            img_path: 图像路径，在此模型中不使用
            
        返回:
            hr_feats: 高分辨率特征
            pseudo_gt: 伪真值特征
            auxiliary_info: 辅助信息（此处为None）
        """
        # 1. 获取低分辨率特征
        lr_feats = self.vfm_model(x)
        
        # 2. 使用LiFT模型生成高分辨率特征
        hr_feats = self.lift_model(lr_feats, x)
        
        # 3. 通过bicubic上采样获取伪真值（双三次插值特征）
        # 这里我们使用简单的双三次插值作为伪真值
        pseudo_gt = F.interpolate(lr_feats, size=hr_feats.shape[2:], mode='bicubic', align_corners=False)
        
        # 4. 在实际应用中，可以使用更高质量的伪真值
        # 例如，根据alpha混合原始特征和插值特征
        if self.alpha < 1.0:
            pseudo_gt = self.alpha * pseudo_gt + (1 - self.alpha) * hr_feats.detach()
        
        return hr_feats, pseudo_gt, None
    
    def configure_optimizers(self):
        """配置优化器，只训练lift_model的参数"""
        if self.optimizer is not None:
            return self.optimizer
            
        # 默认使用Adam优化器
        return torch.optim.Adam(self.lift_model.parameters(), lr=1e-4)