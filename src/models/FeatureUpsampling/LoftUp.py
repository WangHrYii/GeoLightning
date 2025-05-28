import torch
import torch.nn.functional as F
from src.models.FeatureUpsampling.BaseModel import FeatureUpsamplingBase
from src.models.backbones.loft import apply_mask_optimization


# 第一阶段训练器：基于类别无关掩码的训练
class LoftUpStage1Trainer(FeatureUpsamplingBase):
    def __init__(self, 
                 vfm_model,
                 upsampler,
                 sam_model,
                 optimizer=None,
                 alpha: float = 0.8,
                 vfm_input_size: int = 224,
                 original_img_size: int = 1024):
        """
        LoftUp第一阶段训练器：基于类别无关掩码的训练
        
        参数:
            vfm_model: 基础视觉模型，如DINOv2
            upsampler: LoftUp上采样器
            sam_model: SAM2模型
            optimizer: 优化器
            alpha: 掩码优化系数，控制掩码优化的程度
            vfm_input_size: VFM输入尺寸(默认224)
            original_img_size: 原始图像尺寸(默认1024)
        """
        super().__init__(optimizer=optimizer, alpha=alpha)
        self.save_hyperparameters(ignore=['vfm_model', 'upsampler', 'sam_model'])
        
        # 模型组件
        self.vfm_model = vfm_model
        self.upsampler = upsampler
        self.sam_model = sam_model

        # 参数
        self.vfm_input_size = vfm_input_size
        self.original_img_size = original_img_size

        # 冻结基础模型
        for param in self.vfm_model.parameters():
            param.requires_grad = False
        for param in self.sam_model.parameters():
            param.requires_grad = False
            
    def forward(self, x, original_img=None, img_path=None):
        """
        前向传播函数
        
        参数:
            x: 输入图像 [B, C, H, W]，已调整为VFM尺寸(224x224)
            original_img: 原始高分辨率图像 [B, C, H, W]
            img_path: 图像路径列表，用于掩码缓存
            
        返回:
            hr_feats: 高分辨率特征
            f_mask_bicubic: 掩码优化后的双三次插值特征（伪真值）
            batch_masks_info: 掩码信息列表
        """
        # 1. 获取低分辨率特征
        lr_feats = self.vfm_model(x)
        
        # 2. 通过upsampler获取高分辨率特征
        hr_feats = self.upsampler(lr_feats, x)
        
        # 3. 通过bicubic上采样获取F0（双三次插值特征）
        f_bicubic = F.interpolate(lr_feats, size=x.shape[2:], mode='bicubic', align_corners=False)
        
        # 4. 获取SAM masks并计算掩码优化的双三次插值特征
        batch_size = x.shape[0]
        batch_masks_info = []
        
        # 为批次中的每张图片获取SAM掩码
        for b in range(batch_size):
            # 获取原始图像和路径
            current_original_img = original_img[b] if original_img is not None else x[b]
            current_img_path = img_path[b] if img_path is not None else f"img_{b}"
            
            # 获取调整尺寸后的SAM掩码（按面积排序）
            masks_info = self.sam_model.get_sam_masks(current_original_img, current_img_path)
            batch_masks_info.append(masks_info)
        
        # 5. 应用掩码优化策略（最小掩码优先：先处理小掩码，后处理大掩码）
        f_mask_bicubic = apply_mask_optimization(f_bicubic, batch_masks_info, self.alpha)
        
        return hr_feats, f_mask_bicubic, batch_masks_info
    
    def configure_optimizers(self):
        # 创建优化器，只训练upsampler的参数
        trainable_params = self.upsampler.parameters()
        optimizer = self.optimizer(params=trainable_params)
        
        # Define learning rate scheduler
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=self.trainer.max_epochs
            ),
            'interval': 'epoch',
            'name': 'lr'
        }
        
        return [optimizer], [scheduler]

