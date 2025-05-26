import torch
import torch.nn as nn
from lightning import LightningModule
from typing import Any, Dict, Optional, Tuple, Union, List, Callable
from torch.optim import AdamW, Optimizer
import torchmetrics
from torch.optim.lr_scheduler import ReduceLROnPlateau


class BaseRegressionTask(LightningModule):
    def __init__(
        self,
        model:                 nn.Module,
        learning_rate:         float = 1e-3,
        weight_decay:          float = 1e-4,
        lr_scheduler_patience: int = 5,
        lr_scheduler_factor:   float = 0.5,
        metrics:   Optional[List[torchmetrics.Metric]] = None,
        optimizer: Optional[Callable[[List[torch.nn.Parameter]], Optimizer]] = None,
        scheduler: Optional[Callable[[Optimizer], Dict[str, Any]]] = None,
        loss_fn:   Optional[nn.Module] = None,
    ):
        """
        基础回归任务类
        
        参数:
            model: 完整的模型，负责从输入生成预测
            learning_rate: 学习率
            weight_decay:  权重衰减
            lr_scheduler_patience: 学习率调度器的耐心值
            lr_scheduler_factor:   学习率调度器的减少因子
            metrics:   可选的评估指标列表
            optimizer: 可选的优化器创建函数，接收模型参数并返回优化器实例
            scheduler: 可选的调度器创建函数，接收优化器并返回调度器配置字典
            loss_fn:   可选的损失函数，默认为MSE损失
        """
        super().__init__()
        self.save_hyperparameters(ignore=['model', 'optimizer', 'scheduler', 'loss_fn', 'metrics'])
        
        self.model = model                     # 核心模型，后续继承BaseRegressionTask时，只需要传入模型，专注于模型的实现
        self.learning_rate = learning_rate
        self.weight_decay  = weight_decay
        self.lr_scheduler_patience = lr_scheduler_patience
        self.lr_scheduler_factor = lr_scheduler_factor
        
        # 自定义优化器和调度器
        self.optimizer_fn = optimizer
        self.scheduler_fn = scheduler
        
        # 损失函数，允许自定义
        self.loss_fn = loss_fn if loss_fn is not None else nn.MSELoss()
        
        # 指标
        self.metrics = nn.ModuleList(metrics or [])  # 使用ModuleList以便自动追踪
    
    def forward(self, *args, **kwargs):
        """
        模型前向传播，直接委托给底层模型
        """
        return self.model(*args, **kwargs)
    
    def _compute_loss(self, batch: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算损失
        
        参数:
            batch: 输入批次数据
            
        返回:
            loss: 损失值
            y_pred: 预测值
            y_true: 真实值
        """
        # 这里假设batch是(x, y)的元组，如果有不同的数据结构，可以在子类中重写此方法
        x, y_true = batch
        y_pred = self(x)   # 直接self，调用forward方法
        loss   = self.loss_fn(y_pred, y_true)
        return loss, y_pred, y_true
    
    def on_fit_start(self):
        """
        训练开始时，确保所有指标都在正确的设备上
        """
        device = next(self.parameters()).device
        self.metrics = self.metrics.to(device)
    
    def training_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        训练步骤
        
        参数:
            batch: 输入批次数据
            batch_idx: 批次索引
            
        返回:
            包含训练损失的字典
        """
        loss, y_pred, y_true = self._compute_loss(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # 确保预测值和真实值在同一设备上
        device = loss.device
        y_pred = y_pred.to(device)
        y_true = y_true.to(device)
        
        # 记录其他指标
        for metric in self.metrics:
            metric = metric.to(device)  # 确保指标在正确的设备上
            value = metric(y_pred, y_true)
            self.log(f"train_{metric.__class__.__name__}", value, on_step=False, on_epoch=True)
            
        return {"loss": loss}
    
    def validation_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        验证步骤
        
        参数:
            batch: 输入批次数据
            batch_idx: 批次索引
            
        返回:
            包含验证损失的字典
        """
        loss, y_pred, y_true = self._compute_loss(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # 确保预测值和真实值在同一设备上
        device = loss.device
        y_pred = y_pred.to(device)
        y_true = y_true.to(device)
        
        # 记录其他指标
        for metric in self.metrics:
            metric = metric.to(device)  # 确保指标在正确的设备上
            value = metric(y_pred, y_true)
            self.log(f"val_{metric.__class__.__name__}", value, on_step=False, on_epoch=True)
            
        return {"val_loss": loss}
    
    def test_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        测试步骤
        
        参数:
            batch: 输入批次数据
            batch_idx: 批次索引
            
        返回:
            包含测试损失的字典
        """
        loss, y_pred, y_true = self._compute_loss(batch)
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        
        # 确保预测值和真实值在同一设备上
        device = loss.device
        y_pred = y_pred.to(device)
        y_true = y_true.to(device)
        
        # 记录其他指标
        for metric in self.metrics:
            metric = metric.to(device)  # 确保指标在正确的设备上
            value = metric(y_pred, y_true)
            self.log(f"test_{metric.__class__.__name__}", value, on_step=False, on_epoch=True)
            
        return {"test_loss": loss}
    
    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        """
        预测步骤
        
        参数:
            batch: 输入批次数据
            batch_idx: 批次索引
            dataloader_idx: 数据加载器索引
            
        返回:
            预测值
        """
        # 这里假设batch是(x, y)的元组，对于纯预测可能需要调整
        if isinstance(batch, tuple) and len(batch) >= 1:
            x = batch[0]
        else:
            x = batch
        return self(x)
    
    def _default_optimizer(self) -> Optimizer:
        """
        默认优化器
        
        返回:
            AdamW 优化器
        """
        return AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
    
    def _default_scheduler(self, optimizer: Optimizer) -> Dict[str, Any]:
        """
        默认学习率调度器
        
        参数:
            optimizer: 优化器
            
        返回:
            调度器配置字典
        """
        return {
            "scheduler": ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=self.lr_scheduler_factor,
                patience=self.lr_scheduler_patience,
                verbose=True,
            ),
            "monitor": "val_loss",
            "interval": "epoch",
            "frequency": 1,
        }
    
    def configure_optimizers(self) -> Union[Optimizer, Dict[str, Any]]:
        """
        配置优化器和学习率调度器
        
        返回:
            包含优化器和学习率调度器的配置
        """
        # 使用自定义优化器或默认优化器
        if self.optimizer_fn is not None:
            optimizer = self.optimizer_fn(self.parameters())
        else:
            optimizer = self._default_optimizer()
        
        # 如果没有调度器，只返回优化器
        if self.scheduler_fn is None and self.lr_scheduler_patience is None:
            return optimizer
        
        # 使用自定义调度器或默认调度器
        if self.scheduler_fn is not None:
            scheduler_config = self.scheduler_fn(optimizer)
        else:
            scheduler_config = self._default_scheduler(optimizer)
        
        return {"optimizer": optimizer, "lr_scheduler": scheduler_config}


