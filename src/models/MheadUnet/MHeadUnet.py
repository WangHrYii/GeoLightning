""" Swin Transformer based Multi-head UNet """
from typing import Any, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from PIL import Image

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
        self.calculate_f1  = F1Score(num_classes=net.n_classes, task="multiclass")      # 计算F1
        self.calculaye_iou = JaccardIndex(num_classes=net.n_classes, task="multiclass") # 计算IoU

        # 回归精度
        self.calculate_mse = MeanSquaredError()    # 计算MSE
        self.calculate_mae = MeanAbsoluteError()   # 计算MAE
        self.calculate_r2  = R2Score()             # 计算R2

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, mask, regression, _ = batch
        logits, reg = self.net(x)
        reg = reg.squeeze(1)

        loss_seg = F.cross_entropy(logits, mask)

        # 筛选出mask不为0的位置
        mask_non_zero = mask.flatten() != 0
        mask_zero = mask.flatten() == 0

        reg_selected = reg.flatten()[mask_non_zero]
        regression_selected = regression.flatten()[mask_non_zero]

        reg_selected_zero = reg.flatten()[mask_zero]
        regression_selected_zero = regression.flatten()[mask_zero]

        loss_reg_1 = F.mse_loss(reg_selected, regression_selected) + F.l1_loss(reg_selected, regression_selected)
        loss_reg_0 = F.mse_loss(reg_selected_zero, regression_selected_zero) + F.l1_loss(reg_selected_zero, regression_selected_zero)

        loss_reg = loss_reg_1 + loss_reg_0

        f1_score = self.calculate_f1(logits, mask)
        iou_score = self.calculaye_iou(logits, mask)

        loss = loss_seg + loss_reg
        self.log("train_seg_loss", loss_seg, prog_bar=True)
        self.log("train_reg_loss", loss_reg, prog_bar=True)

        self.log("train_f1", f1_score, prog_bar=True)
        self.log("train_iou", iou_score, prog_bar=True)

        self.log("train_loss", loss, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, mask, regression, _ = batch
        logits, reg = self.net(x)
        reg = reg.squeeze(1)

        loss_seg = F.cross_entropy(logits, mask)
        loss_reg = F.mse_loss(reg, regression) + F.l1_loss(reg, regression)

        f1_score = self.calculate_f1(logits, mask)
        iou_score = self.calculaye_iou(logits, mask)

        # mse = self.calculate_mse(reg, regression)
        # mae = self.calculate_mae(reg, regression)
        
        # # 计算r2之前需要先Expected both prediction and target to be 1D or 2D tensors
        # reg = reg.view(-1)
        # regression = regression.view(-1)
        # r2 = self.calculate_r2(reg, regression)

        loss = loss_seg + loss_reg
        self.log("val_seg_loss", loss_seg, prog_bar=True)
        self.log("val_reg_loss", loss_reg, prog_bar=True)
        self.log("val_loss", loss, prog_bar=True)

        # 计算mask==1的像素点的mse和mae以及r2
        # 筛选出mask不为0的位置
        mask_non_zero = mask.flatten() != 0
        reg_selected = reg.flatten()[mask_non_zero]
        regression_selected = regression.flatten()[mask_non_zero]

        mse = self.calculate_mse(reg_selected, regression_selected)
        mae = self.calculate_mae(reg_selected, regression_selected)
        r2 = self.calculate_r2(reg_selected, regression_selected)

        self.log("val_mse", mse, prog_bar=True)
        self.log("val_mae", mae, prog_bar=True)
        self.log("val_r2", r2, prog_bar=True)

        self.log("val_f1", f1_score, prog_bar=True)
        self.log("val_iou", iou_score, prog_bar=True)

        self.log("val_f1+val_r2", f1_score + r2, prog_bar=False)
        
        return loss

    def test_step(self, batch, batch_idx):
        x, mask, regression, name = batch
        logits, reg = self.net(x)
        reg = reg.squeeze(1)

        # 保存预测结果，回归和分类结果都保存，保存为图片
        # 保存分类结果
        # logits = torch.argmax(logits, dim=1)
        # logits = logits.cpu().numpy()
        # if not os.path.exists("test_results"):
        #     os.makedirs("test_results")
        # for i in range(logits.shape[0]):
        #     Image.fromarray(logits[i].astype(np.uint8)).save(f"test_results/{name[i]}_seg.png", cmap="gray")
        # # 保存回归结果
        # reg = reg.cpu().numpy()
        # for i in range(reg.shape[0]):
        #     Image.fromarray(reg[i].astype(np.uint8)).save(f"test_results/{name[i]}_reg.png", cmap="gray")

        loss_seg = F.cross_entropy(logits, mask)
        loss_reg = F.mse_loss(reg, regression) + F.l1_loss(reg, regression)

        f1_score = self.calculate_f1(logits, mask)
        iou_score = self.calculaye_iou(logits, mask)

        mse = self.calculate_mse(reg, regression)
        mae = self.calculate_mae(reg, regression)
        # 计算r2之前需要先Expected both prediction and target to be 1D or 2D tensors
        reg = reg.view(-1)
        regression = regression.view(-1)
        r2 = self.calculate_r2(reg, regression)

        loss = loss_seg + loss_reg
        self.log("test_seg_loss", loss_seg, prog_bar=True)
        self.log("test_reg_loss", loss_reg, prog_bar=True)
        self.log("test_loss", loss, prog_bar=True)

        self.log("test_mse", mse, prog_bar=True)
        self.log("test_mae", mae, prog_bar=True)
        self.log("test_r2", r2, prog_bar=True)

        self.log("test_f1", f1_score, prog_bar=True)
        self.log("test_iou", iou_score, prog_bar=True)
        
        return loss

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_f1",
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
        patch, coords = batch                # coords是patch的坐标, [batch, 2]
        logits, reg = self.net(patch)        # logits [batch, n_classes, h, w], reg [batch, 1, h, w]
        logits = torch.argmax(logits, dim=1) # 将logits转换为分类结果, 运算结束后logits的维度为[batch, h, w]
        return  (logits, reg), coords
