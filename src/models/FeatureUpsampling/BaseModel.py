import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from typing import Dict, Optional, Any
import wandb
from lightning import LightningModule
from tools.vis import visualize_batch_results

class FeatureUpsamplingBase(LightningModule):
    """
    特征上采样基类，统一处理训练、验证、可视化等通用逻辑。
    具体模型只需要继承该基类并重写forward方法。
    """
    def __init__(self, optimizer=None, alpha: float = 0.8):
        """
        初始化特征上采样基类
        
        参数:
            optimizer: 优化器
            alpha: 掩码优化系数
        """
        super().__init__()
        self.save_hyperparameters(ignore=['base_model', 'upsampler', 'sam_model'])
        
        # 优化器
        self.optimizer = optimizer
        
        # 参数
        self.alpha = alpha
        
        # 用于记录和可视化的变量
        self.example_input = None
        self.train_step_outputs = []
        self.val_step_outputs = []
    
    def forward(self, x, original_img=None, img_path=None):
        """
        前向传播函数，子类必须重写该方法
        
        参数:
            x: 输入图像 [B, C, H, W]
            original_img: 原始高分辨率图像 [B, C, H, W]
            img_path: 图像路径列表
            
        返回:
            hr_feats: 高分辨率特征
            pseudo_gt: 伪真值特征
            auxiliary_info: 辅助信息（如mask信息等）
        """
        raise NotImplementedError("子类必须实现forward方法")
    
    def training_step(self, batch, batch_idx):
        """训练步骤"""
        # 获取输入图像和原始高分辨率图像
        x = batch['image']
        original_img = batch['original_image'] if 'original_image' in batch else None
        img_path = batch['path'] if 'path' in batch else None

        # 保存第一个批次作为示例
        if self.example_input is None:
            self.example_input = x.clone()
               
        # 前向传播
        hr_feats, pseudo_gt, auxiliary_info = self(x, original_img, img_path)
        
        # 计算损失：L = ||F^_HR - F_pseudo_gt||_2
        loss = F.mse_loss(hr_feats, pseudo_gt)
        self.log('train_loss_step', loss, prog_bar=True)
                
        # 保存结果用于epoch结束时的处理
        self.train_step_outputs.append({
            'loss': loss.detach(),
            'original_image': original_img if batch_idx == 0 else None,
            'hr_feats': hr_feats.detach() if batch_idx == 0 else None,
            'pseudo_gt': pseudo_gt.detach() if batch_idx == 0 else None,
            'auxiliary_info': auxiliary_info if batch_idx == 0 else None
        })
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """验证步骤"""
        # 获取输入图像和原始高分辨率图像
        x = batch['image']
        original_img = batch['original_image'] if 'original_image' in batch else None
        img_path = batch['path'] if 'path' in batch else None
        
        # 前向传播
        hr_feats, pseudo_gt, auxiliary_info = self(x, original_img, img_path)
        
        # 计算损失
        loss = F.mse_loss(hr_feats, pseudo_gt)
        self.log('val_loss_step', loss, prog_bar=True)
        
        # 保存结果用于epoch结束时的处理
        self.val_step_outputs.append({
            'loss': loss.detach(),
            'original_image': original_img if batch_idx == 0 else None,
            'hr_feats': hr_feats.detach() if batch_idx == 0 else None,
            'pseudo_gt': pseudo_gt.detach() if batch_idx == 0 else None,
            'auxiliary_info': auxiliary_info if batch_idx == 0 else None
        })
        
        return loss
    
    def on_train_epoch_end(self):
        """训练epoch结束时的处理"""
        # 计算平均损失
        avg_loss = torch.stack([x['loss'] for x in self.train_step_outputs]).mean()
        self.log('train_loss', avg_loss, prog_bar=True)

        # 收集批次中的信息用于可视化
        self._visualize_outputs(self.train_step_outputs, "训练")
        
        # 清空记录
        self.train_step_outputs = []
    
    def on_validation_epoch_end(self):
        """验证epoch结束时的处理"""
        # 计算平均损失
        avg_loss = torch.stack([x['loss'] for x in self.val_step_outputs]).mean()
        self.log('val_loss', avg_loss, prog_bar=True)
        
        # 收集批次中的信息用于可视化
        self._visualize_outputs(self.val_step_outputs, "验证")
        
        # 清空记录
        self.val_step_outputs = []
    
    def _visualize_outputs(self, outputs, prefix=""):
        """
        可视化输出结果
        
        参数:
            outputs: 包含训练或验证输出的列表
            prefix: 图像标题前缀
        """
        # 收集批次中的原始图像、高分辨率特征和掩码优化特征
        original_imgs = []
        hr_feats_list = []
        pseudo_gt_list = []
        auxiliary_info_list = []
        
        # 从输出中提取数据
        for output in outputs:
            if 'original_image' in output and output['original_image'] is not None:
                original_imgs.append(output['original_image'])
            if 'hr_feats' in output and output['hr_feats'] is not None:
                hr_feats_list.append(output['hr_feats'])
            if 'pseudo_gt' in output and output['pseudo_gt'] is not None:
                pseudo_gt_list.append(output['pseudo_gt'])
            if 'auxiliary_info' in output and output['auxiliary_info'] is not None:
                auxiliary_info_list.append(output['auxiliary_info'])

        if original_imgs and hr_feats_list and pseudo_gt_list:
            # 转换为批量张量
            original_imgs_tensor = torch.stack(original_imgs)
            hr_feats_tensor = torch.stack(hr_feats_list)
            pseudo_gt_tensor = torch.stack(pseudo_gt_list)
            
            # 生成批量可视化
            combined_images = visualize_batch_results(
                original_imgs_tensor, 
                auxiliary_info_list,
                hr_feats_tensor, 
                pseudo_gt_tensor,
                title=f"Epoch {self.current_epoch} {prefix}"
            )
            
            # 记录到Wandb和保存到本地
            output_dir = os.path.join(self.logger.save_dir if hasattr(self.logger, "save_dir") else "outputs", f"epoch_{self.current_epoch}")
            os.makedirs(output_dir, exist_ok=True)
            
            for i, img in enumerate(combined_images):
                if hasattr(self.logger, "experiment") and hasattr(self.logger.experiment, "log"):
                    self.logger.experiment.log({f"{prefix}_vis_{i}": wandb.Image(img)})
                img_path = os.path.join(output_dir, f"{prefix}_vis_{i}.png")
                img.save(img_path)
    
    def configure_optimizers(self):
        """配置优化器，如果没有自定义则使用默认的Adam优化器"""
        if self.optimizer is not None:
            return self.optimizer
        
        # 默认使用Adam优化器
        return torch.optim.Adam(self.parameters(), lr=1e-4) 