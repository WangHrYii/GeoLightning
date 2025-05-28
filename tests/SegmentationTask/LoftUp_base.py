# src/models/backbones/loftup.py
import torch
import torch.nn as nn
from lightning import LightningModule
from typing import Dict, Optional, List, Any
import wandb
import os
from src.models.backbones.dinov2 import DINOv2
from src.models.backbones.loft import LoftUp, UpsamplerwithChannelNorm, ChannelNorm

class LoftUpBase(LightningModule):
    """LOFTUP基础模型"""
    
    def __init__(
        self,
        vfm_model_name: str,
        vfm_checkpoint_path: str,
        loftup_checkpoint_path: str,
        feature_dim: int = 384
    ):
        super().__init__()
        self.save_hyperparameters()
        self.vfm_model_name = vfm_model_name
        self.vfm_checkpoint_path = vfm_checkpoint_path
        self.loftup_checkpoint_path = loftup_checkpoint_path
        
        # 加载预训练的LOFTUP模型
        self.vfm = DINOv2(self.vfm_model_name)
        self.vfm.load_state_dict(torch.load(self.vfm_checkpoint_path))
        self.vfm.eval()
        self.loftup = self._load_pretrained(self.loftup_checkpoint_path, feature_dim)
        self.loftup.eval()
        
        # 冻结backbone参数
        for param in self.vfm.parameters():
            param.requires_grad = False
        for param in self.loftup.parameters():
            param.requires_grad = False
        
        
        # 用于记录和可视化的变量
        self.train_step_outputs = []
        self.val_step_outputs = []
    
    def _load_pretrained(self, pretrained_path: str, n_dim, lr_pe_type="sine", lr_size=16) -> nn.Module:
        """
        加载LoftUp检查点
        
        参数:
            upsampler_path (str): 检查点路径
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


    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播"""
        lr_feats = self.vfm.get_intermediate_layers(x, reshape=True)[0]
        hr_feats = self.loftup(lr_feats, x)
        
        return {"hr_feats": hr_feats,"lr_feats": lr_feats}
    
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """训练步骤"""
        x = batch['image']
        outputs = self(x)
        
        # 计算损失
        loss = self.compute_loss(outputs, batch)
        
        # 记录指标
        self.log('train_loss_step', loss, prog_bar=True)
        
        # 保存结果用于epoch结束时的处理
        self.train_step_outputs.append({
            'loss': loss.detach(),
            'outputs': outputs,
            'batch': batch,
            'image': x.detach() if batch_idx == 0 else None
        })
        
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """验证步骤"""
        x = batch['image']
        outputs = self(x)
        
        # 计算损失
        loss = self.compute_loss(outputs, batch)
        
        # 记录指标
        self.log('val_loss_step', loss, prog_bar=True)
        
        # 保存结果用于epoch结束时的处理
        self.val_step_outputs.append({
            'loss': loss.detach(),
            'outputs': outputs,
            'batch': batch,
            'image': x.detach() if batch_idx == 0 else None
        })
        
        return loss
    
    def compute_loss(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算损失函数"""
        raise NotImplementedError("子类必须实现compute_loss方法")
    
    def on_train_epoch_end(self):
        """训练epoch结束时的处理"""
        # 计算平均损失
        avg_loss = torch.stack([x['loss'] for x in self.train_step_outputs]).mean()
        self.log('train_loss', avg_loss, prog_bar=True)
        
        # 可视化
        self._visualize_outputs(self.train_step_outputs, "训练")
        
        # 清空记录
        self.train_step_outputs = []
    
    def on_validation_epoch_end(self):
        """验证epoch结束时的处理"""
        # 计算平均损失
        avg_loss = torch.stack([x['loss'] for x in self.val_step_outputs]).mean()
        self.log('val_loss', avg_loss, prog_bar=True)
        
        # 可视化
        self._visualize_outputs(self.val_step_outputs, "验证")
        
        # 清空记录
        self.val_step_outputs = []
    
    def _visualize_outputs(self, outputs: List[Dict], prefix: str):
        """可视化输出结果"""
        if not outputs or outputs[0]['image'] is None:
            return
        
        image = outputs[0]['image']
        
        if hasattr(self.logger, "experiment") and hasattr(self.logger.experiment, "log"):
            self.logger.experiment.log({
                f"{prefix}_images": wandb.Image(image)
            })
    
