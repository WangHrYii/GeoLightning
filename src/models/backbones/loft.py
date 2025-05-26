import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .featup_layers.layers import ChannelNorm, LayerNorm, MinMaxScaler, ImplicitFeaturizer, CATransformer
import os

class LoftUp(nn.Module):
    """
    LoftUp: 基于掩码优化的特征上采样器
    
    使用傅里叶特征作为输入，通过与LR特征进行交叉注意力，输出HR特征。
    支持正弦和可学习位置编码两种模式。
    
    参数:
        dim (int): 输出特征维度
        color_feats (bool): 是否使用颜色特征，默认为True
        n_freqs (int): 傅里叶特征的频率数量，默认为20
        num_heads (int): 注意力头数量，默认为4
        num_layers (int): Transformer层数，默认为2
        num_conv_layers (int): 卷积层数量，默认为1
        lr_size (int): 低分辨率特征大小，默认为16
        lr_pe_type (str): 低分辨率位置编码类型('sine'或'learnable')，默认为'sine'
    """
    def __init__(self, dim, color_feats=True, n_freqs=20, num_heads=4, num_layers=2, num_conv_layers=1, lr_size=16, lr_pe_type="sine"):
        super(LoftUp, self).__init__()

        if color_feats:
            start_dim = 5 * n_freqs * 2 + 3
        else:
            start_dim = 2 * n_freqs * 2
        
        num_patches = lr_size * lr_size
        self.lr_pe_type = lr_pe_type
        if self.lr_pe_type == "sine":
            self.lr_pe = ImplicitFeaturizer(color_feats=False, n_freqs=5, learn_bias=True)
            self.lr_pe_dim = 2 * 5 * 2
        elif self.lr_pe_type == "learnable":
            self.lr_pe = nn.Parameter(torch.randn(1, num_patches, dim))
            self.lr_pe_dim = dim

        self.fourier_feat = torch.nn.Sequential(
                                MinMaxScaler(), # 缩放到[-0.5, 0.5]
                                ImplicitFeaturizer(color_feats, n_freqs=n_freqs, learn_bias=True), # 傅里叶特征
                            )
        if self.lr_pe_type == "sine": # LR PE is concatenated to LR
            self.first_conv = torch.nn.Sequential(
                                ChannelNorm(start_dim),
                                nn.Conv2d(start_dim, dim+self.lr_pe_dim, kernel_size=3, padding=1),
                                nn.BatchNorm2d(dim+self.lr_pe_dim),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(dim+self.lr_pe_dim, dim+self.lr_pe_dim, kernel_size=3, padding=1),
                                nn.BatchNorm2d(dim+self.lr_pe_dim),
                                nn.ReLU(inplace=True),
                                )


            self.final_conv = torch.nn.Sequential(
                nn.Conv2d(dim+self.lr_pe_dim, dim, kernel_size=1),
                LayerNorm(dim),
            )

            self.ca_transformer = CATransformer(dim+self.lr_pe_dim, depth=num_layers, heads=num_heads, dim_head=dim//num_heads, mlp_dim=dim, dropout=0.)
        elif self.lr_pe_type == "learnable": # LR PE is added to LR
            self.first_conv = torch.nn.Sequential(
                                ChannelNorm(start_dim),
                                nn.Conv2d(start_dim, dim, kernel_size=3, padding=1),
                                nn.BatchNorm2d(dim),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(dim, dim, kernel_size=3, padding=1),
                                nn.BatchNorm2d(dim),
                                nn.ReLU(inplace=True),
                                )
            self.final_conv = LayerNorm(dim)
            self.ca_transformer = CATransformer(dim, depth=num_layers, heads=num_heads, dim_head=dim//num_heads, mlp_dim=dim, dropout=0.)

    def forward(self, lr_feats, img):
        """
        前向传播函数
        
        参数:
            lr_feats (Tensor): 低分辨率特征 [B, C, H, W]
            img (Tensor): 输入图像 [B, C, H, W]
            
        返回:
            Tensor: 上采样后的高分辨率特征 [B, dim, H, W]
        """
        # Step 1: Extract Fourier features from the input image
        x = self.fourier_feat(img) # Output shape: (B, dim, H, W)
        b, c, h, w = x.shape

        ## Resize and add LR feats to x? 
        x = self.first_conv(x) # [B, dim, H, W] -> [B, dim+self.lr_pe_dim, H, W]
    
        # Reshape for attention (B, C, H, W) -> (B, H*W, C)
        b, c, h, w = x.shape
        x = x.flatten(2).permute(0, 2, 1)  # (B, H*W, C)

        # Step 2: Process LR features for keys and values
        b, c_lr, h_lr, w_lr = lr_feats.shape

        if self.lr_pe_type == "sine":
            lr_pe = self.lr_pe(lr_feats) # [B, dim, H, W] -> [B, self.lr_pe_dim, H, W]
            lr_feats_with_pe = torch.cat([lr_feats, lr_pe], dim=1) # [B, dim+self.lr_pe_dim, H, W]
            lr_feats_with_pe = lr_feats_with_pe.flatten(2).permute(0, 2, 1) # [B, H*W, dim+self.lr_pe_dim]
        elif self.lr_pe_type == "learnable":
            lr_feats = lr_feats.flatten(2).permute(0, 2, 1) # (B, H*W, C)
            if lr_feats.shape[1] != self.lr_pe.shape[1]:
                len_pos_old = int(math.sqrt(self.lr_pe.shape[1]))
                pe = self.lr_pe.reshape(1, len_pos_old, len_pos_old, c_lr).permute(0, 3, 1, 2)
                pe = F.interpolate(pe, size=(h_lr, w_lr), mode='bicubic', align_corners=False)
                pe = pe.reshape(1, c_lr, h_lr*w_lr).permute(0, 2, 1)
                lr_feats_with_pe = lr_feats + pe
            else:
                lr_feats_with_pe = lr_feats + self.lr_pe
        x = self.ca_transformer(x, lr_feats_with_pe)   

        # Reshape back to (B, C, H, W)
        x = x.permute(0, 2, 1).reshape(b, c, h, w)

        return self.final_conv(x)

class UpsamplerwithChannelNorm(nn.Module):
    """
    带有通道归一化的上采样器包装类
    
    参数:
        upsampler (nn.Module): 上采样器模型
        channelnorm (nn.Module): 通道归一化模块
    """
    def __init__(self, upsampler, channelnorm):
        super(UpsamplerwithChannelNorm, self).__init__()
        self.upsampler = upsampler
        self.channelnorm = channelnorm

    def forward(self, lr_feats, img):
        """
        前向传播函数
        
        参数:
            lr_feats (Tensor): 低分辨率特征 [B, C, H, W]
            img (Tensor): 输入图像 [B, C, H, W]
            
        返回:
            Tensor: 上采样后的高分辨率特征 [B, dim, H, W]
        """
        lr_feats = self.channelnorm(lr_feats)
        return self.upsampler(lr_feats, img)



class Bilinear(torch.nn.Module):
    """双线性上采样模块"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, feats, img):
        """
        前向传播函数
        
        参数:
            feats (Tensor): 需要上采样的特征 [B, C, h, w]
            img (Tensor): 目标图像 [B, C, H, W]，用于确定输出尺寸
            
        返回:
            Tensor: 双线性上采样后的特征 [B, C, H, W]
        """
        _, _, h, w = img.shape
        return F.interpolate(feats, (h, w), mode="bilinear", align_corners=False)

def get_upsampler(upsampler, dim, lr_size=16, n_freqs=20, cfg=None, cat_lr_feats=True, lr_pe_type="sine"):
    """
    获取上采样器实例
    
    参数:
        upsampler (str): 上采样器类型，如'loftup'或'bilinear'
        dim (int): 特征维度
        lr_size (int): 低分辨率特征大小，默认为16
        n_freqs (int): 傅里叶特征的频率数量，默认为20
        cfg (dict): 其他配置参数，默认为None
        cat_lr_feats (bool): 是否拼接低分辨率特征，默认为True
        lr_pe_type (str): 低分辨率位置编码类型，默认为'sine'
        
    返回:
        nn.Module: 上采样器实例
    """
    if upsampler == "loftup":
        return LoftUp(dim, n_freqs=n_freqs, lr_size=lr_size, lr_pe_type=lr_pe_type)
    elif upsampler == "bilinear":
        return Bilinear()
    else:
        raise ValueError(f"不支持的上采样器类型: {upsampler}")


def apply_mask_optimization(f_bicubic, masks_info, alpha=0.8):
    """
    应用掩码优化策略，使用排序后的掩码迭代优化特征
    遵循最小掩码优先原则：每个像素最终使用覆盖它的最小面积掩码的特征
    
    参数:
        f_bicubic (Tensor): 双三次插值特征 [B, C, H, W]
        masks_info (list): 掩码信息列表，按面积从大到小排序
        alpha (float): 掩码优化系数，控制掩码优化的程度，默认为0.8
        
    返回:
        Tensor: 掩码优化后的特征 [B, C, H, W]
    """
    # 初始化掩码优化后的特征为双三次插值特征的副本
    f_mask_bicubic = f_bicubic.clone()
    
    # 检查masks_info是否为空
    if not masks_info:
        return f_mask_bicubic
    
    batch_size = f_bicubic.shape[0]
    for b in range(batch_size):
        # 提取当前批次图片的特征
        current_f_bicubic = f_bicubic[b:b+1]
        pixel_processed = torch.zeros_like(current_f_bicubic[:, 0:1]) # 记录每个像素是否被处理过

        # 处理当前批次的掩码
        if b < len(masks_info) and masks_info[b]:
            current_masks = masks_info[b]
               
            # 反转掩码顺序，从小到大处理（最小掩码优先）
            reversed_masks = list(reversed(current_masks))
            
            for mask_info in reversed_masks:
                # 检查mask_info是否是字典并包含'mask'键
                if not isinstance(mask_info, dict) or 'mask' not in mask_info:
                    continue
                    
                m = mask_info['mask'].to(f_bicubic.device)
                
                # 扩展掩码以匹配特征维度
                mask_expanded = m.view(1, 1, *m.shape).expand_as(current_f_bicubic)
                # 只处理未处理过的像素
                effective_mask = mask_expanded * (1 - pixel_processed)
                
                # 如果当前掩码有效
                if effective_mask.sum() > 0:
                    # 计算掩码区域的平均特征
                    masked_feats = current_f_bicubic * effective_mask
                    mask_sum = effective_mask.sum(dim=(2, 3), keepdim=True)
                    
                    if torch.sum(mask_sum) > 0:
                        # 计算掩码内像素的平均特征值
                        mask_mean = (masked_feats.sum(dim=(2, 3), keepdim=True) / (mask_sum + 1e-8))
                        mask_mean_expanded = mask_mean.expand_as(current_f_bicubic)
                        
                        # 应用掩码优化公式: F_Mask-Bicubic[m] = α·F_Bicubic[m]¯ + (1-α)·F_Bicubic[m]
                        f_mask_bicubic[b:b+1] = torch.where(
                            effective_mask > 0,
                            alpha * mask_mean_expanded + (1 - alpha) * current_f_bicubic,
                            f_mask_bicubic[b:b+1]
                        )
                        
                        # 更新已处理标记
                        pixel_processed = torch.where(
                            effective_mask[:, 0:1] > 0,
                            torch.ones_like(pixel_processed),
                            pixel_processed
                        )  
    return f_mask_bicubic

def load_loftup_checkpoint(pretrained_path: str, n_dim: int, lr_pe_type: str = "sine", lr_size: int = 16) -> nn.Module:
    """
    加载LoftUp检查点
    
    参数:
        pretrained_path (str): 检查点路径
        n_dim (int): 特征维度
        lr_pe_type (str): 低分辨率位置编码类型，默认为'sine'
        lr_size (int): 低分辨率特征大小，默认为16
        
    返回:
        nn.Module: 加载好权重的UpsamplerwithChannelNorm模块
    """
    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(f"预训练模型文件不存在: {pretrained_path}")
        
    channelnorm = ChannelNorm(n_dim)
    upsampler = LoftUp(n_dim, lr_pe_type=lr_pe_type, lr_size=lr_size)
    
    # 加载检查点
    ckpt_weight = torch.load(pretrained_path, map_location='cpu')
    
    # 处理不同格式的检查点
    if 'state_dict' in ckpt_weight:
        ckpt_weight = ckpt_weight['state_dict']
    
    # 提取通道归一化权重
    channelnorm_checkpoint = {k: v for k, v in ckpt_weight.items() if 'model.1' in k} 
    channelnorm_checkpoint = {k.replace('model.1.', ''): v for k, v in channelnorm_checkpoint.items()}
    
    # 提取上采样器权重
    upsampler_ckpt_weight = {k: v for k, v in ckpt_weight.items() if k.startswith('upsampler')}
    upsampler_ckpt_weight = {k.replace('upsampler.', ''): v for k, v in upsampler_ckpt_weight.items()}
    
    # 加载权重
    upsampler.load_state_dict(upsampler_ckpt_weight)
    channelnorm.load_state_dict(channelnorm_checkpoint)
    
    return UpsamplerwithChannelNorm(upsampler, channelnorm)
