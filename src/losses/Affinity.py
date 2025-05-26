import torch
import torch.nn as nn
import torch.nn.functional as F

# 亲和力矩阵损失函数
class AffinityMatrixLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        计算两个特征图之间的亲和力矩阵损失
        
        参数:
            x: 预测特征 [B, C, H, W]
            y: 目标特征 [B, C, H, W]
            
        返回:
            loss: 亲和力矩阵损失
        """
        # 将特征展平为 [B, C, H*W]
        B, C, H, W = x.shape
        x_flat = x.view(B, C, -1)
        y_flat = y.view(B, C, -1)
        
        # 计算亲和力矩阵 [B, H*W, H*W]
        x_affinity = torch.bmm(x_flat.transpose(1, 2), x_flat)
        y_affinity = torch.bmm(y_flat.transpose(1, 2), y_flat)
        
        # 归一化亲和力矩阵
        x_affinity = F.normalize(x_affinity, p=2, dim=2)
        y_affinity = F.normalize(y_affinity, p=2, dim=2)
        
        # 计算损失
        loss = F.mse_loss(x_affinity, y_affinity)
        return loss