""" Swin Transformer based Multi-head UNet """
from typing import Any, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from PIL import Image
import wandb
import matplotlib.pyplot as plt

from src.models.MheadUnet.unet_parts import Down, Up, OutConv, DoubleConv

from lightning import LightningModule
from torchmetrics import (
    JaccardIndex, 
    F1Score,
    MeanSquaredError,
    MeanAbsoluteError, 
    R2Score
)
from src.utils import RankedLogger


log = RankedLogger(__name__, rank_zero_only=True)

class UnetEncoder(nn.Module):
    """
    UNet 编码器, 返回四个尺度的特征图
    """
    def __init__(self, n_channels):
        super(UnetEncoder, self).__init__()
        self.n_channels = n_channels

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        return x1, x2, x3, x4, x5


class MheadUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super().__init__()
        self.n_classes = n_classes


        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)


        # 分割头
        self.up1_1 = Up(1024, 512, bilinear)
        self.up1_2 = Up(512, 256, bilinear)
        self.up1_3 = Up(256, 128, bilinear)
        self.up1_4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

        # 回归头
        self.up2_1 = Up(1024, 512, bilinear)
        self.up2_2 = Up(512, 256, bilinear)
        self.up2_3 = Up(256, 128, bilinear)
        self.up2_4 = Up(128, 64, bilinear)
        self.outr = OutConv(64, 1)
    
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # 分割头
        seg = self.up1_1(x5, x4)
        seg = self.up1_2(seg, x3)
        seg = self.up1_3(seg, x2)
        seg = self.up1_4(seg, x1)
        seg = self.outc(seg)

        # 回归头
        reg = self.up2_1(x5, x4)
        reg = self.up2_2(reg, x3)
        reg = self.up2_3(reg, x2)
        reg = self.up2_4(reg, x1)
        reg = self.outr(reg)

        return seg, reg


class MheadUNetLM(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
    ):
        """Initialize a `MheadUNetLM`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.compile = compile
        
        # 分割精度
        self.calculate_f1 = F1Score(num_classes=net.n_classes, task="binary")      # 计算F1
        self.calculate_iou = JaccardIndex(num_classes=net.n_classes, task="binary") # 计算IoU

        # 回归精度
        self.calculate_mse = MeanSquaredError()    # 计算MSE
        self.calculate_mae = MeanAbsoluteError()   # 计算MAE
        self.calculate_r2 = R2Score()              # 计算R2

        # 用于存储验证样本
        self.validation_samples = None

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, mask, regression, _ = batch
        mask = mask.unsqueeze(1)  # 增加一个维度
        regression = regression.unsqueeze(1)  # 增加一个维度
        logits, reg = self.net(x)

        # 分割损失
        bce_loss = F.binary_cross_entropy_with_logits(logits, mask.float())
        dice_loss = self._dice_loss(logits, mask)
        iou_loss = self._iou_loss(logits, mask)
        seg_loss = bce_loss + dice_loss + iou_loss

        # 高度损失 - 只计算mask>0的部分
        mask_bool = mask > 0
        
        # 检查是否有足够的样本
        if mask_bool.sum() > 1:  # 至少需要2个样本计算R2
            l1_loss = F.l1_loss(reg[mask_bool], regression[mask_bool])
            l2_loss = F.mse_loss(reg[mask_bool], regression[mask_bool])
            
            # 计算R2损失 - 只计算mask>0的部分
            ss_res = torch.sum((regression[mask_bool] - reg[mask_bool]) ** 2)
            ss_tot = torch.sum((regression[mask_bool] - regression[mask_bool].mean()) ** 2)
            r2_loss = 1 - (ss_res / (ss_tot + 1e-8))
            
            # 组合高度损失
            height_loss = l1_loss + l2_loss
        else:
            # 如果没有足够的样本，使用零损失
            l1_loss = torch.tensor(0.0, device=reg.device)
            l2_loss = torch.tensor(0.0, device=reg.device)
            r2_loss = torch.tensor(0.0, device=reg.device)
            height_loss = torch.tensor(0.0, device=reg.device)

        # 总损失
        loss = seg_loss + height_loss

        # 计算指标
        f1_score = self.calculate_f1(logits, mask)
        iou_score = self.calculate_iou(logits, mask)
        
        # 回归指标 - 只计算mask>0的部分
        if mask_bool.sum() > 1:
            mse = F.mse_loss(reg[mask_bool], regression[mask_bool])
            mae = F.l1_loss(reg[mask_bool], regression[mask_bool])
            
            # 展平张量以计算R2 - 只计算mask>0的部分
            reg_flat = reg[mask_bool].view(-1)
            regression_flat = regression[mask_bool].view(-1)
            
            # 确保有足够的样本计算R2
            if len(reg_flat) > 1:
                r2 = self.calculate_r2(reg_flat, regression_flat)
            else:
                r2 = torch.tensor(0.0, device=reg.device)
        else:
            mse = torch.tensor(0.0, device=reg.device)
            mae = torch.tensor(0.0, device=reg.device)
            r2 = torch.tensor(0.0, device=reg.device)

        # 记录学习率
        for i, param_group in enumerate(self.trainer.optimizers[0].param_groups):
            self.log(f"lr/group_{i}", param_group['lr'], on_step=True)

        # 记录损失和指标
        self.log("train/seg_loss", seg_loss, prog_bar=True)
        self.log("train/height_loss", height_loss, prog_bar=True)
        self.log("train/total_loss", loss, prog_bar=True)
        self.log("train/f1", f1_score, prog_bar=True)
        self.log("train/iou", iou_score, prog_bar=True)
        self.log("train/mse", mse, prog_bar=True)
        self.log("train/mae", mae, prog_bar=True)
        self.log("train/r2", r2, prog_bar=True)
        self.log("train/height_max", reg[mask_bool].max() if mask_bool.any() else torch.tensor(0.0, device=reg.device), on_step=False, on_epoch=True, reduce_fx=torch.max)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, mask, regression, _ = batch
        mask = mask.unsqueeze(1)  # 增加一个维度
        regression = regression.unsqueeze(1)  # 增加一个维度
        logits, reg = self.net(x)

        # 分割损失
        bce_loss = F.binary_cross_entropy_with_logits(logits, mask.float())
        dice_loss = self._dice_loss(logits, mask)
        iou_loss = self._iou_loss(logits, mask)
        seg_loss = bce_loss + dice_loss + iou_loss

        # 高度损失 - 只计算mask>0的部分
        mask_bool = mask > 0
        
        # 检查是否有足够的样本
        if mask_bool.sum() > 1:  # 至少需要2个样本计算R2
            l1_loss = F.l1_loss(reg[mask_bool], regression[mask_bool])
            l2_loss = F.mse_loss(reg[mask_bool], regression[mask_bool])
            
            # 计算R2损失 - 只计算mask>0的部分
            ss_res = torch.sum((regression[mask_bool] - reg[mask_bool]) ** 2)
            ss_tot = torch.sum((regression[mask_bool] - regression[mask_bool].mean()) ** 2)
            r2_loss = 1 - (ss_res / (ss_tot + 1e-8))
            
            # 组合高度损失
            height_loss = l1_loss + l2_loss + (1 - r2_loss) * 10
        else:
            # 如果没有足够的样本，使用零损失
            l1_loss = torch.tensor(0.0, device=reg.device)
            l2_loss = torch.tensor(0.0, device=reg.device)
            r2_loss = torch.tensor(0.0, device=reg.device)
            height_loss = torch.tensor(0.0, device=reg.device)

        # 总损失
        loss = seg_loss + height_loss

        # 计算指标
        f1_score = self.calculate_f1(logits, mask)
        iou_score = self.calculate_iou(logits, mask)
        
        # 回归指标 - 只计算mask>0的部分
        if mask_bool.sum() > 1:
            mse = F.mse_loss(reg[mask_bool], regression[mask_bool])
            mae = F.l1_loss(reg[mask_bool], regression[mask_bool])
            
            # 展平张量以计算R2 - 只计算mask>0的部分
            reg_flat = reg[mask_bool].view(-1)
            regression_flat = regression[mask_bool].view(-1)
            
            # 确保有足够的样本计算R2
            if len(reg_flat) > 1:
                r2 = self.calculate_r2(reg_flat, regression_flat)
            else:
                r2 = torch.tensor(0.0, device=reg.device)
        else:
            mse = torch.tensor(0.0, device=reg.device)
            mae = torch.tensor(0.0, device=reg.device)
            r2 = torch.tensor(0.0, device=reg.device)

        # 记录损失和指标
        self.log("val/seg_loss", seg_loss, prog_bar=True)
        self.log("val/height_loss", height_loss, prog_bar=True)
        self.log("val/total_loss", loss, prog_bar=True)
        self.log("val/f1", f1_score, prog_bar=True)
        self.log("val/iou", iou_score, prog_bar=True)
        self.log("val/mse", mse, prog_bar=True)
        self.log("val/mae", mae, prog_bar=True)
        self.log("val/r2", r2, prog_bar=True)
        self.log("val/height_max", reg[mask_bool].max() if mask_bool.any() else torch.tensor(0.0, device=reg.device), on_epoch=True, reduce_fx=torch.max)

        # 存储第一个batch的样本用于可视化
        if batch_idx == 0:
            self.validation_samples = {
                'images': x.cpu(),
                'seg_gt': mask.cpu(),
                'height_gt': regression.cpu(),
                'seg_pred': torch.sigmoid(logits).cpu(),
                'height_pred': reg.cpu()
            }
        
        return loss

    def on_validation_epoch_end(self):
        if self.validation_samples is not None:
            # 随机选择3张图片进行可视化
            indices = torch.randperm(len(self.validation_samples['images']))[:3]
            
            for i,idx in enumerate(indices):
                # 创建可视化
                fig, axes = plt.subplots(1, 5, figsize=(20, 4))
                
                # 原始图像 - 只取前三个波段(RGB)
                img = self.validation_samples['images'][idx].permute(1, 2, 0).numpy()
                img = img[..., :3].astype(np.uint8)
                axes[0].imshow(img)
                axes[0].set_title('Original Image (RGB)')
                axes[0].axis('off')
                
                # 分割真值
                seg_gt = self.validation_samples['seg_gt'][idx].squeeze(0).numpy()
                axes[1].imshow(seg_gt, cmap='gray')
                axes[1].set_title('Segmentation GT')
                axes[1].axis('off')
                
                # 高度真值
                height_gt = self.validation_samples['height_gt'][idx].squeeze(0).numpy()
                im = axes[2].imshow(height_gt, cmap='viridis')
                axes[2].set_title('Height GT')
                axes[2].axis('off')
                plt.colorbar(im, ax=axes[2])
                
                # 分割预测
                seg_pred = self.validation_samples['seg_pred'][idx].squeeze(0).numpy()
                axes[3].imshow(seg_pred, cmap='gray')
                axes[3].set_title('Segmentation Pred')
                axes[3].axis('off')
                
                # 高度预测
                height_pred = self.validation_samples['height_pred'][idx].squeeze(0).numpy()
                im = axes[4].imshow(height_pred, cmap='viridis')
                axes[4].set_title('Height Pred')
                axes[4].axis('off')
                plt.colorbar(im, ax=axes[4])
                
                plt.tight_layout()
                
                # 记录到wandb
                wandb.log({
                    f"validation_sample_{i}": wandb.Image(fig),
                    "epoch": self.current_epoch
                })
                plt.close()
            
            # 清理样本
            self.validation_samples = None

    def test_step(self, batch, batch_idx):
        x, mask, regression, name = batch
        mask = mask.unsqueeze(1)  # 增加一个维度
        regression = regression.unsqueeze(1)  # 增加一个维度
        logits, reg = self.net(x)

        # 分割损失
        bce_loss = F.binary_cross_entropy_with_logits(logits, mask.float())
        dice_loss = self._dice_loss(logits, mask)
        iou_loss = self._iou_loss(logits, mask)
        seg_loss = bce_loss + dice_loss + iou_loss

        # 高度损失 - 只计算mask>0的部分
        mask_bool = mask > 0
        
        # 检查是否有足够的样本
        if mask_bool.sum() > 1:  # 至少需要2个样本计算R2
            l1_loss = F.l1_loss(reg[mask_bool], regression[mask_bool])
            l2_loss = F.mse_loss(reg[mask_bool], regression[mask_bool])
            
            # 计算R2损失 - 只计算mask>0的部分
            ss_res = torch.sum((regression[mask_bool] - reg[mask_bool]) ** 2)
            ss_tot = torch.sum((regression[mask_bool] - regression[mask_bool].mean()) ** 2)
            r2_loss = 1 - (ss_res / (ss_tot + 1e-8))
            
            # 组合高度损失
            height_loss = l1_loss + l2_loss + (1 - r2_loss) * 10
        else:
            # 如果没有足够的样本，使用零损失
            l1_loss = torch.tensor(0.0, device=reg.device)
            l2_loss = torch.tensor(0.0, device=reg.device)
            r2_loss = torch.tensor(0.0, device=reg.device)
            height_loss = torch.tensor(0.0, device=reg.device)

        # 总损失
        loss = seg_loss + height_loss

        # 计算指标
        f1_score = self.calculate_f1(logits, mask)
        iou_score = self.calculate_iou(logits, mask)
        
        # 回归指标 - 只计算mask>0的部分
        if mask_bool.sum() > 1:
            mse = F.mse_loss(reg[mask_bool], regression[mask_bool])
            mae = F.l1_loss(reg[mask_bool], regression[mask_bool])
            
            # 展平张量以计算R2 - 只计算mask>0的部分
            reg_flat = reg[mask_bool].view(-1)
            regression_flat = regression[mask_bool].view(-1)
            
            # 确保有足够的样本计算R2
            if len(reg_flat) > 1:
                r2 = self.calculate_r2(reg_flat, regression_flat)
            else:
                r2 = torch.tensor(0.0, device=reg.device)
        else:
            mse = torch.tensor(0.0, device=reg.device)
            mae = torch.tensor(0.0, device=reg.device)
            r2 = torch.tensor(0.0, device=reg.device)

        # 记录损失和指标
        self.log("test/seg_loss", seg_loss, prog_bar=True)
        self.log("test/height_loss", height_loss, prog_bar=True)
        self.log("test/total_loss", loss, prog_bar=True)
        self.log("test/f1", f1_score, prog_bar=True)
        self.log("test/iou", iou_score, prog_bar=True)
        self.log("test/mse", mse, prog_bar=True)
        self.log("test/mae", mae, prog_bar=True)
        self.log("test/r2", r2, prog_bar=True)
        self.log("test/height_max", reg[mask_bool].max() if mask_bool.any() else torch.tensor(0.0, device=reg.device), on_epoch=True, reduce_fx=torch.max)
        
        return loss

    def _dice_loss(self, pred, target):
        smooth = 1e-5
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        return 1 - ((2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth))

    def _iou_loss(self, pred, target):
        smooth = 1e-5
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum() - intersection
        return 1 - ((intersection + smooth) / (union + smooth))

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/total_loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def predict_step(self, batch, batch_idx):
        patch, coords = batch
        logits, reg = self.net(patch)
        logits = torch.sigmoid(logits)
        return (logits, reg), coords
