# from networks.networks.UNet import *
# from networks.networks.ALLSpark import *
# from networks.networks.UperNet import UPerNet
# from networks.networks.hrocr import HrOcr
# from networks.networks.ResUNet_dense import res_unet_dense
# from networks.networks.PAN import PAN
# from networks.networks.swinunet import SwinUNet
# from networks.networks.mask2former import Mask2Former
# import torch

import torch
import lightning as L
from torch import nn
import torch.nn.functional as F
import torchmetrics
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from src.registries import MODEL_REGISTRY
from src.registries import NETWORK_REGISTRY
from src.registries import LOSS_REGISTRY
from src.registries import OPTIMIZER_REGISTRY
from src.registries import SCHEDULER_REGISTRY

from src.utils import RankedLogger
import swanlab

import hydra
import logging
import numpy as np
import random
from torch.utils.data import DataLoader
from src.utils.colorvis import ColorVis
import cv2
from tabulate import tabulate

log = RankedLogger(__name__, rank_zero_only=True)



# 注册模型
@MODEL_REGISTRY.register("SemanticSegmentationLightning")
class SemanticSegmentationLightning(L.LightningModule):
    def __init__(self, cfg):
        super(SemanticSegmentationLightning, self).__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg
        self.model = self.create_model(self.cfg.net)
        if self.cfg.net_pretrained_file is not None:
            try:
                state_dict = torch.load(self.cfg.net_pretrained_file)
                self.model.load_state_dict(state_dict, strict=True)
                log.info(f"成功加载预训练权重: {self.cfg.net_pretrained_file}")
            except Exception as e:
                log.error(f"加载预训练权重失败: {e}")
                raise

    def setup(self, stage=None):
        self.ignore_index = self.cfg.ignore_index
        self.class_info = self.trainer.datamodule.train_dataloader().dataset.get_class_info()
        self.class_names = [info['name'] for info in self.class_info]
        self.num_classes = len(self.class_names)
        self.img2color = ColorVis(self.trainer.datamodule.train_dataloader().dataset.get_color_table())

        # 获取类别权重，如果未设置则使用全1权重
        if hasattr(self.cfg.loss, 'class_weight') and self.cfg.loss.class_weight is not None:
            self.class_weights = torch.tensor(self.cfg.loss.class_weight, device=self.device)
        else:
            # 创建全1权重，ignore类别设为0
            self.class_weights = torch.ones(self.num_classes, device=self.device)
            # 处理多个ignore类别
            ignore_indices = self.ignore_index
            if isinstance(ignore_indices, (list, tuple)):
                # 如果是列表，将所有ignore类别的权重设为0
                for idx in ignore_indices:
                    self.class_weights[idx] = 0.0
            else:
                # 如果是单个值，直接设置
                self.class_weights[ignore_indices] = 0.0

        # 定义指标 (average=None, except for precision and recall)
        metrics = {
            "accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes, ignore_index=self.ignore_index, average=None),
            "mIoU": torchmetrics.JaccardIndex(task="multiclass", num_classes=self.num_classes, ignore_index=self.ignore_index, average=None),
            "f1": torchmetrics.F1Score(task="multiclass", num_classes=self.num_classes, ignore_index=self.ignore_index, average=None),
            "precision": torchmetrics.Precision(task="multiclass", num_classes=self.num_classes, ignore_index=self.ignore_index, average=None), 
            "recall": torchmetrics.Recall(task="multiclass", num_classes=self.num_classes, ignore_index=self.ignore_index, average=None),
        }

        # 使用 MetricCollection
        self.train_metrics = torchmetrics.MetricCollection(metrics)
        self.val_metrics = torchmetrics.MetricCollection({k: m.clone() for k, m in metrics.items()})

        if stage == "fit" or stage is None:
            train_dataset = self.trainer.datamodule.train_dataloader().dataset
            val_dataset = self.trainer.datamodule.val_dataloader().dataset

            # 获取训练集和验证集的样本个数
            num_train_samples = len(train_dataset)
            num_val_samples = len(val_dataset)

            # 生成随机索引
            self.train_indices = random.sample(range(num_train_samples), self.cfg.visualization.num_samples)  # 使用实际的样本个数
            self.val_indices = random.sample(range(num_val_samples), self.cfg.visualization.num_samples)    # 使用实际的样本个数

            # 将选择的样本组成 batch
            self.example_train_batch = torch.utils.data.Subset(train_dataset, self.train_indices)
            self.example_val_batch = torch.utils.data.Subset(val_dataset, self.val_indices)

    # 可视化预测结果
    def visualize_predictions(self, batch, dataset_type="train"):
        # 创建 DataLoader
        subset_loader = DataLoader(batch, batch_size=self.cfg.visualization.num_samples, shuffle=False) # 使用可视化样本数作为batch_size
        # 循环处理每个 batch
        for batch_idx, batch in enumerate(subset_loader):
            images, targets, name = batch['image'].to(self.device), batch['label'].to(self.device), batch['name']
            predictions = self(images)
            predictions = F.softmax(predictions, dim=1)
            predictions = torch.argmax(predictions, dim=1)

            std = self.trainer.datamodule.train_dataset.std_value
            mean = self.trainer.datamodule.train_dataset.mean_value
  
            for i in range(images.shape[0]):
                original_image = data_denormalize(images[i].cpu().numpy(), mean, std, self.cfg.visualization.display_bands).astype(np.uint8)
                # 将 targets 和 predictions 转换为 NumPy 数组
                targets_np = self.img2color.run(targets[i].cpu().numpy())
                predictions_np = self.img2color.run(predictions[i].cpu().numpy())

                # 使用 np.concatenate 拼接图像
                grid_np = np.concatenate([
                    original_image,
                    targets_np,  # targets_np 已经是numpy数组
                    predictions_np, # predictions_np已经是numpy数组
                ], axis=-1)  # 在最后一个维度上拼接 (水平拼接)
                
                resized_np = cv2.resize(grid_np.transpose(1, 2, 0), (128*3, 128), interpolation=cv2.INTER_AREA)
                
                if isinstance(self.logger, swanlab.integration.pytorch_lightning.SwanLabLogger):
                    # 记录图像，添加索引以区分不同的图像
                    image = swanlab.Image(resized_np, caption=name[i])
                    self.logger.experiment.log({f"{dataset_type}/{name[i]}": image})
                else:
                    # 记录图像，添加索引以区分不同的图像
                    self.logger.experiment.add_image(f"{dataset_type}/{name[i]}", resized_np, self.current_epoch, dataformats='CHW')
                
    def create_model(self, cfg):
        # 从注册器获取模型类
        model_class = NETWORK_REGISTRY.get(cfg.net_name)
        if model_class is None:
            supported_models = list(NETWORK_REGISTRY._dict.keys())
            raise ValueError(f"Model '{cfg.net_name}' is not registered. Supported models are: {', '.join(supported_models)}")
        
        backbone_instance = hydra.utils.instantiate(cfg.backbone)

        # 获取模型构造函数的参数
        return model_class(backbone=backbone_instance, num_class=cfg.num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, targets = batch['image'], batch['label']
        outputs = self(images)

        total_loss = 0
        for loss_fn_name, weight in zip(self.cfg.loss.loss_func, self.cfg.loss.loss_weight):
            loss_fn_class = LOSS_REGISTRY.get(loss_fn_name)
            if loss_fn_class is None:
                supported_losses = list(LOSS_REGISTRY.keys())
                raise ValueError(f"Loss function '{loss_fn_name}' is not registered. Supported loss functions are: {', '.join(supported_losses)}")
            loss_fn = loss_fn_class(weight = self.class_weights, ignore_index = self.ignore_index)  # 实例化 loss_fn
            loss = loss_fn(outputs, targets)
            total_loss += weight * loss

        preds = torch.argmax(outputs, dim=1)
        
        # 更新指标
        self.train_metrics(preds, targets)

        # Log
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=self.trainer.datamodule.train_dataloader().dataset.cfg.train_batch_size, sync_dist=True)
        self.log('lr', self.lr_schedulers().get_last_lr()[0], on_step=True, prog_bar=False, batch_size=self.trainer.datamodule.train_dataloader().dataset.cfg.train_batch_size, sync_dist=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch['image'], batch['label']
        outputs = self(images)
        
        total_loss = 0
        for loss_fn_name, weight in zip(self.cfg.loss.loss_func, self.cfg.loss.loss_weight):
            loss_fn_class = LOSS_REGISTRY.get(loss_fn_name)
            if loss_fn_class is None:
                supported_losses = list(LOSS_REGISTRY._dict.keys())
                raise ValueError(f"Loss function '{loss_fn_name}' is not registered. Supported loss functions are: {', '.join(supported_losses)}")
            loss_fn = loss_fn_class(weight=self.class_weights, ignore_index=self.ignore_index)  # 实例化 loss_fn
            loss = loss_fn(outputs, targets)
            total_loss += weight * loss

        preds = torch.argmax(outputs, dim=1)

        # 更新指标 (所有类别一起更新)
        self.val_metrics(preds, targets)

        self.log('val_loss', total_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=self.trainer.datamodule.val_dataloader().dataset.cfg.val_batch_size, sync_dist=True)
        return total_loss

    def on_validation_epoch_end(self):
        log.info(f"epoch = {self.current_epoch}")
        self.log_metrics(self.train_metrics, "train")
        self.log_metrics(self.val_metrics, "val")
        if self.current_epoch % self.cfg.visualization.interval == 0:
            self.visualize_predictions(self.example_train_batch, "train")
            self.visualize_predictions(self.example_val_batch, "val")

    def configure_optimizers(self):
        # 从注册器中加载优化器
        optimizer_config = {
            'lr': self.cfg.optimizer.lr,
            'momentum': self.cfg.optimizer.momentum,
            'weight_decay': self.cfg.optimizer.weight_decay
        }

        optimizer_fn = OPTIMIZER_REGISTRY.get(self.cfg.optimizer.name)
        if optimizer_fn is None:
            supported_optimizer = list(OPTIMIZER_REGISTRY.keys())
            raise ValueError(f"Optimizer '{self.cfg.optimizer.name}' is not registered. Supported optimizers are: {', '.join(supported_optimizer)}")
        optimizer = optimizer_fn(self.model.parameters(), optimizer_config)

        scheduler_config = {
            'max_lr': self.cfg.optimizer.lr,
            'gamma': self.cfg.scheduler.lr_scheduler_gamma,   # 学习率更新因子
            'step_size': self.cfg.scheduler.lr_step_size,     # 对于step_lr，每隔多少个epoch更新一次学习率
            'steps': self.cfg.scheduler.lr_steps,             # 对于multistep_lr，在哪些epoch时更新学习率
            'cos_lr_t': self.cfg.scheduler.cos_lr_t,          # 对于cos_lr或cos_restart，退火周期（epoch数）
            'cos_t_mult': self.cfg.scheduler.cos_t_mult,      # 对于cos_restart，下一个退火周期相对于上一个周期的倍数
            'total_steps': self.trainer.estimated_stepping_batches,  # 对于one_cycle，若未指定则自动计算每个epoch的步数
            'total_epoches': self.trainer.max_epochs,
        }
        scheduler = SCHEDULER_REGISTRY.get(self.cfg.scheduler.lr_scheduler)(optimizer, scheduler_config)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step" if self.cfg.scheduler.lr_scheduler == "one_cycle" else "epoch",
                "frequency": 1
            }
        }

    def log_metrics(self, metrics, prefix):
        computed_metrics = metrics.compute()
        metric_names = list(computed_metrics.keys())  # 获取所有 metric 名称
        log_data = {}  # 存储类别的 metric 值
        average_values = []  # 存储平均值行

        for metric_name, values in computed_metrics.items():
            if values.numel() > 1:
                values_valid = values[values.isfinite()]  # 排除 NaN 和 Inf
                average_value = torch.nanmean(values_valid).item() if values_valid.numel() > 0 else float("nan")
                # self.log(f'{prefix}_{metric_name}/Average', average_value, prog_bar=(metric_name in ["accuracy", "mIoU"]), sync_dist=True)
                self.log(f'{prefix}_{metric_name}/Average', average_value, prog_bar=False, logger=True, sync_dist=True)

                if prefix == "val":
                    average_values.append(f"{average_value:.4f}")  # 记录平均值
                
                for i, name in enumerate(self.class_names):
                    if i == self.ignore_index:  # 忽略 ignore_index 类别
                        continue
                    if torch.isfinite(values[i]):
                        self.log(f'{prefix}_{metric_name}/{name}', values[i], prog_bar=False, logger=True, sync_dist=True)
                        if prefix == "val":
                            log_data.setdefault(name, []).append(f"{values[i].item():.4f}")  # 记录类别的 metric 值
                    elif prefix == "val":
                        log_data.setdefault(name, []).append("N/A")  # 若无效则填充 "N/A"
            else:
                value_str = f"{values.item():.4f}" if torch.isfinite(values) else "N/A"
                self.log(f'{prefix}_{metric_name}', values, prog_bar=False, logger=True, sync_dist=True)
                if prefix == "val":
                    average_values.append(value_str)
                    for name in self.class_names:
                        if name != self.class_names[self.ignore_index]:  # 忽略 ignore_index 类别
                            log_data.setdefault(name, []).append(value_str)

        # 仅在 val 阶段输出表格
        if prefix == "val":
            table_data = [["Average"] + average_values]  # 先加入平均值行
            for name, values in log_data.items():
                table_data.append([name] + values)

            headers = [f"{prefix}_Class"] + metric_names  # 第一列是类别，后续是 metric 名称
            table = tabulate(table_data, headers=headers, tablefmt="grid")
            log.info(f"\n{table}")

        metrics.reset()

def data_denormalize(src, mean, std, display_bands=None):
    """
    Args:
        src: array of shape C*H*W
        mean: list of length C
        std: list of length C
        display_bands: list of band indices to display
    """
    if display_bands is None:
        display_bands = list(range(len(mean)))
    
    dst = src[display_bands].copy()
    for i, band in enumerate(display_bands):
        dst[i] = 255 * (src[band] * std[band] + mean[band])
    return dst.astype(np.uint8)