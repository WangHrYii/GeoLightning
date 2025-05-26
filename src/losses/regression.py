# func: regeression_losses
# Desc: regression loss functions
# Path: src/losses/regression.py
# author: wanghr
# ------------------------------------------------------------------------------------ #
import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaleInvariantLoss(nn.Module):
    """
    缩放不变损失，用于树木高度回归
    
    基于论文：Eigen et al., "Depth Map Prediction from a Single Image using a Multi-Scale Deep Network"
    """
    def __init__(self, lambd=0.5):
        super().__init__()
        self.lambd = lambd
        
    def forward(self, pred, target):
        """
        计算缩放不变损失
        
        Args:
            pred: 预测高度，形状为 [B, H, W]
            target: 真实高度，形状为 [B, H, W]
            mask: 可选掩码，形状为 [B, H, W]，指示哪些像素应该参与损失计算
            
        Returns:
            损失值
        """
        b, h, w = pred.shape
        n = b * h * w
        
        # 计算对数差
        log_diff = torch.log(pred + 1e-6) - torch.log(target + 1e-6)
        
        # 计算损失的第一项：对数差的平方和的均值
        term1 = torch.sum(log_diff**2) / n
        
        # 计算损失的第二项：对数差的和的平方除以n的平方
        term2 = (torch.sum(log_diff) / n)**2
        
        # 计算最终损失
        loss = term1 - self.lambd * term2
        
        return torch.sqrt(loss)


class GradientConsistencyLoss(nn.Module):
    """
    梯度一致性损失，用于保持高度图的平滑性和边界连贯性
    """
    def __init__(self):
        super().__init__()
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        
    def forward(self, pred, target):
        """
        计算梯度一致性损失
        
        Args:
            pred: 预测高度，形状为 [B, H, W]
            target: 真实高度，形状为 [B, H, W]
            mask: 可选掩码，形状为 [B, H, W]，指示哪些像素应该参与损失计算
            
        Returns:
            损失值
        """
        device = pred.device
        dtype = pred.dtype
        
        # 确保滤波器与输入张量有相同的数据类型
        sobel_x = self.sobel_x.to(device).to(dtype)
        sobel_y = self.sobel_y.to(device).to(dtype)
        
        # 确保输入有通道维度
        pred_c = pred
        target_c = target
        
        # 计算梯度
        pred_grad_x = F.conv2d(pred_c, sobel_x, padding=1)
        pred_grad_y = F.conv2d(pred_c, sobel_y, padding=1)
        target_grad_x = F.conv2d(target_c, sobel_x, padding=1)
        target_grad_y = F.conv2d(target_c, sobel_y, padding=1)
        
        # 移除通道维度
        pred_grad_x = pred_grad_x.squeeze(1)
        pred_grad_y = pred_grad_y.squeeze(1)
        target_grad_x = target_grad_x.squeeze(1)
        target_grad_y = target_grad_y.squeeze(1)
        
        # 计算梯度差异
        grad_diff_x = torch.abs(pred_grad_x - target_grad_x)
        grad_diff_y = torch.abs(pred_grad_y - target_grad_y)
        
        b,_, h, w = pred.shape
        n = b * h * w
        
        # 计算最终损失
        loss = (torch.sum(grad_diff_x) + torch.sum(grad_diff_y)) / n
        
        return loss