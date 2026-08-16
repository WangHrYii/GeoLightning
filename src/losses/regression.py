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


class EdgeAwareLoss(nn.Module):
    """
    边缘感知损失：在边缘区域施加更强的约束，使高度图边界更清晰
    """
    def __init__(self, edge_weight=10.0):
        super().__init__()
        self.edge_weight = edge_weight
        # Sobel算子用于边缘检测
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)

    def _compute_edges(self, x):
        """计算边缘图"""
        device = x.device
        dtype = x.dtype
        sobel_x = self.sobel_x.to(device).to(dtype)
        sobel_y = self.sobel_y.to(device).to(dtype)

        grad_x = F.conv2d(x, sobel_x, padding=1)
        grad_y = F.conv2d(x, sobel_y, padding=1)
        edges = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
        return edges

    def forward(self, pred, target):
        """
        Args:
            pred: 预测高度 [B, 1, H, W]
            target: 真实高度 [B, 1, H, W]
        """
        # 计算GT的边缘图作为权重
        target_edges = self._compute_edges(target)
        # 归一化边缘权重到[1, edge_weight]
        edge_weights = 1.0 + (self.edge_weight - 1.0) * (target_edges / (target_edges.max() + 1e-6))

        # 加权L1损失
        weighted_loss = edge_weights * torch.abs(pred - target)
        return weighted_loss.mean()


class MaskedHeightLoss(nn.Module):
    """
    掩码加权高度损失：只在树木区域计算损失，避免背景主导
    """
    def __init__(self, bg_weight=0.1, fg_weight=1.0):
        super().__init__()
        self.bg_weight = bg_weight  # 背景权重
        self.fg_weight = fg_weight  # 前景(树木)权重

    def forward(self, pred, target, mask):
        """
        Args:
            pred: 预测高度 [B, 1, H, W]
            target: 真实高度 [B, 1, H, W]
            mask: 分割掩码 [B, 1, H, W], 1=树木, 0=背景
        """
        # 创建权重图
        weights = torch.where(mask > 0.5, self.fg_weight, self.bg_weight)

        # 加权L1损失
        loss = weights * torch.abs(pred - target)
        return loss.mean()


class LocalContrastLoss(nn.Module):
    """
    局部对比度损失：鼓励模型保持局部高度变化，避免过度平滑
    """
    def __init__(self, kernel_size=5):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

    def forward(self, pred, target):
        """
        Args:
            pred: 预测高度 [B, 1, H, W]
            target: 真实高度 [B, 1, H, W]
        """
        # 计算局部均值
        kernel = torch.ones(1, 1, self.kernel_size, self.kernel_size,
                           device=pred.device, dtype=pred.dtype)
        kernel = kernel / (self.kernel_size ** 2)

        pred_mean = F.conv2d(pred, kernel, padding=self.padding)
        target_mean = F.conv2d(target, kernel, padding=self.padding)

        # 计算局部方差(对比度)
        pred_var = F.conv2d((pred - pred_mean)**2, kernel, padding=self.padding)
        target_var = F.conv2d((target - target_mean)**2, kernel, padding=self.padding)

        # 鼓励预测的局部方差接近GT的局部方差
        contrast_loss = torch.abs(pred_var - target_var).mean()
        return contrast_loss


class SharpHeightLoss(nn.Module):
    """
    组合损失：专门用于生成清晰的高度图

    组合了:
    1. 掩码加权L1损失 - 聚焦树木区域
    2. 边缘感知损失 - 保持边界清晰
    3. 局部对比度损失 - 避免过度平滑
    4. 梯度一致性损失 - 保持结构
    """
    def __init__(
        self,
        edge_weight=5.0,
        contrast_weight=0.5,
        gradient_weight=1.0,
        bg_weight=0.1
    ):
        super().__init__()
        self.edge_loss = EdgeAwareLoss(edge_weight=edge_weight)
        self.contrast_loss = LocalContrastLoss(kernel_size=5)
        self.gradient_loss = GradientConsistencyLoss()
        self.masked_loss = MaskedHeightLoss(bg_weight=bg_weight)

        self.contrast_weight = contrast_weight
        self.gradient_weight = gradient_weight

    def forward(self, pred, target, mask=None):
        """
        Args:
            pred: 预测高度 [B, 1, H, W]
            target: 真实高度 [B, 1, H, W]
            mask: 分割掩码 [B, 1, H, W] (可选)
        """
        # 基础损失
        if mask is not None:
            base_loss = self.masked_loss(pred, target, mask)
        else:
            base_loss = F.l1_loss(pred, target)

        # 边缘损失
        edge_loss = self.edge_loss(pred, target)

        # 对比度损失
        contrast_loss = self.contrast_loss(pred, target)

        # 梯度损失
        grad_loss = self.gradient_loss(pred, target)

        total = (base_loss + edge_loss +
                 self.contrast_weight * contrast_loss +
                 self.gradient_weight * grad_loss)

        return total