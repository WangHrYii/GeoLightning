#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAR-Optical语义分割Lightning模型
基于MCANet架构，集成到PyTorch Lightning框架
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, PolynomialLR, OneCycleLR, ReduceLROnPlateau
import lightning as L
from lightning.pytorch.utilities.types import STEP_OUTPUT
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
import warnings

from src.models.MultiModalSegTask.MCANet import MACANet

# 抑制警告
warnings.filterwarnings("ignore")


class DiceLoss(nn.Module):
    """Dice损失函数，用于语义分割"""
    
    def __init__(self, smooth: float = 1.0, ignore_index: int = -100):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # input: (N, C, H, W), target: (N, H, W)
        input = F.softmax(input, dim=1)
        
        # 确保target是long类型并且在有效范围内
        target = target.long()
        num_classes = input.shape[1]
        
        # 处理ignore_index：将其设置为0，后续用mask排除
        valid_mask = (target != self.ignore_index)
        target_masked = torch.where(valid_mask, target, torch.zeros_like(target))
        
        # 限制target在有效范围内
        target_masked = torch.clamp(target_masked, 0, num_classes - 1)
        
        # 创建one-hot编码
        target_one_hot = F.one_hot(target_masked, num_classes=num_classes).permute(0, 3, 1, 2).float()
        
        # 应用valid_mask
        if self.ignore_index >= 0:
            mask = valid_mask.unsqueeze(1).float()
            input = input * mask
            target_one_hot = target_one_hot * mask
        
        # 计算Dice系数
        intersection = (input * target_one_hot).sum(dim=(2, 3))
        union = input.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class FocalLoss(nn.Module):
    """Focal损失函数，处理类别不平衡"""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, ignore_index: int = -100):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(input, target, ignore_index=self.ignore_index, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class CombinedLoss(nn.Module):
    """组合损失函数：交叉熵 + Dice + Focal"""
    
    def __init__(
        self,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        focal_weight: float = 0.5,
        class_weights: Optional[torch.Tensor] = None,
        ignore_index: int = -100
    ):
        super(CombinedLoss, self).__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.ignore_index = ignore_index
        
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)
        self.dice_loss = DiceLoss(ignore_index=ignore_index)
        self.focal_loss = FocalLoss(ignore_index=ignore_index)
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        ce = self.ce_loss(input, target)
        dice = self.dice_loss(input, target)
        focal = self.focal_loss(input, target)
        
        total_loss = (
            self.ce_weight * ce + 
            self.dice_weight * dice + 
            self.focal_weight * focal
        )
        
        return {
            'total_loss': total_loss,
            'ce_loss': ce,
            'dice_loss': dice,
            'focal_loss': focal
        }


def calculate_metrics(pred: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_index: int = -100) -> Dict[str, float]:
    """计算分割指标"""
    pred = pred.argmax(dim=1)  # (N, H, W)
    
    # 创建有效mask，排除ignore_index
    valid_mask = (target != ignore_index)
    
    metrics = {}
    
    # 总体准确率（只计算有效像素）
    if valid_mask.sum() > 0:
        correct = ((pred == target) & valid_mask).float()
        accuracy = correct.sum() / valid_mask.sum().float()
        metrics['accuracy'] = accuracy.item()
    else:
        metrics['accuracy'] = 0.0
    
    # 每类IoU和mIoU
    ious = []
    for cls in range(num_classes):
        pred_mask = (pred == cls) & valid_mask
        target_mask = (target == cls) & valid_mask
        
        intersection = (pred_mask & target_mask).float().sum()
        union = (pred_mask | target_mask).float().sum()
        
        if union > 0:
            iou = intersection / union
        else:
            iou = torch.tensor(float('nan'))
        
        ious.append(iou.item())
        metrics[f'iou_class_{cls}'] = iou.item()
    
    # 计算mIoU（忽略nan值）
    valid_ious = [iou for iou in ious if not np.isnan(iou)]
    if valid_ious:
        metrics['miou'] = np.mean(valid_ious)
    else:
        metrics['miou'] = 0.0
    
    return metrics


class SAROpticalSegmentation(L.LightningModule):
    """
    SAR-Optical语义分割Lightning模型
    基于MCANet架构
    """
    
    def __init__(
        self,
        num_classes: int = 6,
        backbone: str = 'ResNet101',
        pretrained: bool = True,
        att_type: Optional[str] = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        optimizer: str = 'adamw',  # 'adamw', 'sgd'
        scheduler: str = 'cosine',  # 'cosine', 'polynomial', 'onecycle'
        max_epochs: int = 100,
        loss_config: Optional[Dict] = None,
        class_weights: Optional[List[float]] = None,
        ignore_index: int = -100,
        monitor_metric: str = 'val_miou'
    ):
        """
        初始化模型
        
        Args:
            num_classes: 分割类别数
            backbone: 主干网络类型
            pretrained: 是否使用预训练权重
            att_type: 注意力机制类型
            learning_rate: 学习率
            weight_decay: 权重衰减
            optimizer: 优化器类型
            scheduler: 学习率调度器类型
            max_epochs: 最大训练轮数
            loss_config: 损失函数配置
            class_weights: 类别权重
            ignore_index: 忽略的标签值
            monitor_metric: 监控的指标
        """
        super().__init__()
        self.save_hyperparameters()
        
        # 模型配置
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer
        self.scheduler_name = scheduler
        self.max_epochs = max_epochs
        self.monitor_metric = monitor_metric
        
        # 创建模型
        self.model = MACANet(
            num_classes=num_classes,
            pretrained=pretrained,
            backbone=backbone,
            att_type=att_type
        )
        
        # 设置损失函数
        if loss_config is None:
            loss_config = {
                'ce_weight': 1.0,
                'dice_weight': 1.0, 
                'focal_weight': 0.5
            }
        
        # 处理类别权重
        if class_weights is not None:
            class_weights = torch.tensor(class_weights, dtype=torch.float32)
        
        self.criterion = CombinedLoss(
            class_weights=class_weights,
            ignore_index=ignore_index,
            **loss_config
        )
        
        # 记录最佳指标
        self.best_miou = 0.0
        
    def forward(self, sar_img: torch.Tensor, optical_img: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.model(sar_img, optical_img)
    
    def _shared_step(self, batch: Dict[str, torch.Tensor], batch_idx: int, stage: str) -> Dict[str, torch.Tensor]:
        """共享的训练/验证/测试步骤"""
        # 提取数据
        sar_img = batch['sar']          # (B, C, H, W)
        optical_img = batch['optical']  # (B, C, H, W)
        target = batch['mask']          # (B, H, W)
        
        # 确保target是long类型
        target = target.long()
        
        # 前向传播
        logits = self.forward(sar_img, optical_img)  # (B, num_classes, H, W)
        
        # 调整target尺寸以匹配logits
        if logits.shape[-2:] != target.shape[-2:]:
            target = F.interpolate(
                target.unsqueeze(1).float(), 
                size=logits.shape[-2:], 
                mode='nearest'
            ).squeeze(1).long()
        
        # 计算损失
        loss_dict = self.criterion(logits, target)
        total_loss = loss_dict['total_loss']
        
        # 计算指标
        metrics = calculate_metrics(logits, target, self.num_classes, ignore_index=self.criterion.ignore_index)
        
        # 记录日志
        self.log(f'{stage}_loss', total_loss, on_step=(stage=='train'), on_epoch=True, prog_bar=True, sync_dist=True)  # sync_dist=True 用于多GPU训练
        self.log(f'{stage}_ce_loss', loss_dict['ce_loss'], on_epoch=True, sync_dist=True)
        self.log(f'{stage}_dice_loss', loss_dict['dice_loss'], on_epoch=True, sync_dist=True)
        self.log(f'{stage}_focal_loss', loss_dict['focal_loss'], on_epoch=True, sync_dist=True)
        self.log(f'{stage}_accuracy', metrics['accuracy'], on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f'{stage}_miou', metrics['miou'], on_epoch=True, prog_bar=True, sync_dist=True)
        
        # 记录每类IoU
        for cls in range(self.num_classes):
            if not np.isnan(metrics[f'iou_class_{cls}']):
                self.log(f'{stage}_iou_class_{cls}', metrics[f'iou_class_{cls}'], on_epoch=True, sync_dist=True)
        
        return {
            'loss': total_loss,
            'logits': logits.detach(),
            'target': target,
            'metrics': metrics
        }
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> STEP_OUTPUT:
        """训练步骤"""
        return self._shared_step(batch, batch_idx, 'train')
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> STEP_OUTPUT:
        """验证步骤"""
        return self._shared_step(batch, batch_idx, 'val')
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> STEP_OUTPUT:
        """测试步骤"""
        return self._shared_step(batch, batch_idx, 'test')
    
    def on_validation_epoch_end(self) -> None:
        """验证轮次结束时的处理"""
        # 获取当前验证指标
        current_miou = self.trainer.callback_metrics.get('val_miou', 0.0)
        
        if isinstance(current_miou, torch.Tensor):
            current_miou = current_miou.item()
        
        # 更新最佳指标
        if current_miou > self.best_miou:
            self.best_miou = current_miou
        
        self.log('best_miou', self.best_miou, on_epoch=True, prog_bar=True, sync_dist=True)
    
    def configure_optimizers(self) -> Dict[str, Any]:
        """配置优化器和学习率调度器"""
        # 设置不同的学习率
        backbone_params = []
        decoder_params = []
        
        for name, param in self.model.named_parameters():
            if 'encoder' in name:
                backbone_params.append(param)
            else:
                decoder_params.append(param)
        
        # 创建优化器
        if self.optimizer_name.lower() == 'adamw':
            optimizer = AdamW([
                {'params': backbone_params, 'lr': self.learning_rate * 0.1},  # backbone用较小学习率
                {'params': decoder_params, 'lr': self.learning_rate}
            ], weight_decay=self.weight_decay)
        elif self.optimizer_name.lower() == 'sgd':
            optimizer = SGD([
                {'params': backbone_params, 'lr': self.learning_rate * 0.1},
                {'params': decoder_params, 'lr': self.learning_rate}
            ], momentum=0.9, weight_decay=self.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")
        
        # 创建学习率调度器
        if self.scheduler_name.lower() == 'cosine':
            scheduler = CosineAnnealingLR(optimizer, T_max=self.max_epochs, eta_min=1e-6)
        elif self.scheduler_name.lower() == 'polynomial':
            scheduler = PolynomialLR(optimizer, total_iters=self.max_epochs, power=0.9)
        elif self.scheduler_name.lower() == 'onecycle':
            # 需要知道总的step数，这里使用一个估计值
            total_steps = self.max_epochs * 100  # 假设每个epoch有100个batch
            scheduler = OneCycleLR(optimizer, max_lr=self.learning_rate, total_steps=total_steps)
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step'  # OneCycleLR需要每个step更新
                }
            }
        elif self.scheduler_name.lower() == 'reduceonplateau':
            scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'monitor': self.monitor_metric
                }
            }
        else:
            return optimizer
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'
            }
        }
    
    def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """预测步骤"""
        sar_img = batch['sar']
        optical_img = batch['optical']
        
        logits = self.forward(sar_img, optical_img)
        predictions = F.softmax(logits, dim=1)
        
        return predictions

    def load_state_dict(self, state_dict, strict=True):
        """
        重写load_state_dict方法以处理不兼容的键
        """
        # 过滤掉不应该在模型状态中的键
        filtered_state_dict = {}
        for key, value in state_dict.items():
            # 跳过criterion相关的键
            if not key.startswith('criterion.'):
                filtered_state_dict[key] = value
            else:
                print(f"跳过不兼容的键: {key}")
        
        return super().load_state_dict(filtered_state_dict, strict=strict)
