# src/models/ClassificationTask/LoftUpClassifier.py
import torch
import torch.nn as nn
from typing import Dict, List, Optional
from .LoftUp_base import LoftUpBase
import wandb
import numpy as np
from sklearn.metrics import f1_score, jaccard_score
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T


class LoftUpSegmentation(LoftUpBase):
    """基于LOFTUP的分类任务"""
    
    def __init__(
        self,
        num_classes: int = 6,
        feature_dim: int = 384,
        classifier_dims: List[int] = [256, 128],
        dropout_rate: float = 0.2,
        optimizer: Optional[torch.optim.Optimizer] = None,
        vfm_model_name: str = "dino_vits14",
        vfm_checkpoint_path: str = "pretrained/dino_vits14.pth",
        loftup_checkpoint_path: str = "pretrained/loftup_vits14.pth",
    ):
        super().__init__(
            vfm_model_name=vfm_model_name,
            vfm_checkpoint_path=vfm_checkpoint_path,
            loftup_checkpoint_path=loftup_checkpoint_path,
            feature_dim=feature_dim
        )

        self.optimizer = optimizer
        
        # 构建分割头
        classifier_layers = []
        prev_dim = feature_dim
        
        for dim in classifier_dims:
            classifier_layers.extend([
                nn.Conv2d(prev_dim, dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(dim),
                nn.ReLU(),
                nn.Dropout2d(dropout_rate)
            ])
            prev_dim = dim
        
        classifier_layers.append(nn.Conv2d(prev_dim, num_classes, kernel_size=1))
        self.classifier = nn.Sequential(*classifier_layers)
        
        # 初始化分类头权重
        self._init_classifier_weights()
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss()

    def _init_classifier_weights(self):
        """初始化分类头权重"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播"""
        outputs = super().forward(x)
        hr_feats = outputs['hr_feats']
        lr_feats = outputs['lr_feats']
        logits = self.classifier(hr_feats)    
        
        return {
            'logits': logits,
            'lr_feats': lr_feats,
            'hr_feats': hr_feats
        }
    
    def compute_metrics(self, logits: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        """计算分割指标"""
        preds = torch.argmax(logits, dim=1)
        
        # 转换为numpy数组进行计算
        preds_np = preds.cpu().numpy()
        y_np = y.cpu().numpy()
        
        # 类别名称映射
        class_names = {
            0: "Impervious surfaces",
            1: "Building",
            2: "Low vegetation",
            3: "Tree",
            4: "Car",
            5: "Clutter/background"
        }
        
        # 计算每个类别的指标
        class_metrics = {}
        for class_idx, class_name in class_names.items():
            # 计算该类别的F1分数
            class_f1 = f1_score(
                y_np.flatten() == class_idx,
                preds_np.flatten() == class_idx,
                average='binary'
            )
            
            # 计算该类别的IoU
            class_iou = jaccard_score(
                y_np.flatten() == class_idx,
                preds_np.flatten() == class_idx,
                average='binary'
            )
            
            # 计算该类别的准确率
            class_acc = (preds_np == class_idx)[y_np == class_idx].mean() if (y_np == class_idx).any() else 0.0
            
            # 记录到wandb
            if hasattr(self.logger, "experiment") and hasattr(self.logger.experiment, "log"):
                self.logger.experiment.log({
                    f"{class_name}/f1": class_f1,
                    f"{class_name}/iou": class_iou,
                    f"{class_name}/acc": class_acc
                })
            
            class_metrics[f'{class_name}_f1'] = class_f1
            class_metrics[f'{class_name}_iou'] = class_iou
            class_metrics[f'{class_name}_acc'] = class_acc
        
        # 计算整体指标
        mf1 = f1_score(y_np.flatten(), preds_np.flatten(), average='macro')
        iou = jaccard_score(y_np.flatten(), preds_np.flatten(), average='macro')
        acc = (preds == y).float().mean().item()
        
        # 记录整体指标到wandb
        if hasattr(self.logger, "experiment") and hasattr(self.logger.experiment, "log"):
            self.logger.experiment.log({
                "overall/mf1": mf1,
                "overall/iou": iou,
                "overall/acc": acc
            })
        
        return {
            'acc': acc,
            'mf1': mf1,
            'iou': iou,
            **class_metrics  # 添加每个类别的指标
        }
    
    def compute_loss(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算分割损失"""
        logits = outputs['logits']
        y = batch['mask'].squeeze(1).long()  # [B, H, W]

        loss = self.criterion(logits, y)
        metrics = self.compute_metrics(logits, y)
        
        return loss, metrics
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """重写训练步骤以添加指标计算"""
        loss, metrics = self.compute_loss(self(batch['image']), batch)
        
        # 记录指标
        self.log('train_loss_step', loss, prog_bar=True)
        self.log('train_acc_step', metrics['acc'])
        self.log('train_mf1_step', metrics['mf1'])
        self.log('train_iou_step', metrics['iou'])
        
        # 保存结果用于epoch结束时的处理
        self.train_step_outputs.append({
            'loss': loss.detach(),
            'metrics': metrics,
            'outputs': self(batch['image']),
            'batch': batch,
            'image': batch['image'].detach() if batch_idx == 0 else None
        })
        
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """重写验证步骤以添加指标计算"""
        loss, metrics = self.compute_loss(self(batch['image']), batch)
        
        # 记录指标
        self.log('val_loss_step', loss, prog_bar=True)
        self.log('val_acc_step', metrics['acc'])
        self.log('val_mf1_step', metrics['mf1'])
        self.log('val_iou_step', metrics['iou'])
        
        # 保存结果用于epoch结束时的处理
        self.val_step_outputs.append({
            'loss': loss.detach(),
            'metrics': metrics,
            'outputs': self(batch['image']),
            'batch': batch,
            'image': batch['image'].detach() if batch_idx == 0 else None
        })
        
        return loss
    
    def on_train_epoch_end(self):
        """重写epoch结束处理以添加指标计算"""
        # 计算平均指标
        avg_loss = torch.stack([x['loss'] for x in self.train_step_outputs]).mean()
        avg_acc = np.mean([x['metrics']['acc'] for x in self.train_step_outputs])
        avg_mf1 = np.mean([x['metrics']['mf1'] for x in self.train_step_outputs])
        avg_iou = np.mean([x['metrics']['iou'] for x in self.train_step_outputs])
        
        # 记录指标
        self.log('train_loss', avg_loss, prog_bar=True)
        self.log('train_acc', avg_acc, prog_bar=True)
        self.log('train_mf1', avg_mf1, prog_bar=True)
        self.log('train_iou', avg_iou, prog_bar=True)
        
        # 可视化
        self._visualize_outputs(self.train_step_outputs, "训练")
        
        # 清空记录
        self.train_step_outputs = []
    
    def on_validation_epoch_end(self):
        """重写epoch结束处理以添加指标计算"""
        # 计算平均指标
        avg_loss = torch.stack([x['loss'] for x in self.val_step_outputs]).mean()
        avg_acc = np.mean([x['metrics']['acc'] for x in self.val_step_outputs])
        avg_mf1 = np.mean([x['metrics']['mf1'] for x in self.val_step_outputs])
        avg_iou = np.mean([x['metrics']['iou'] for x in self.val_step_outputs])
        
        # 记录指标
        self.log('val_loss', avg_loss, prog_bar=True)
        self.log('val_acc', avg_acc, prog_bar=True)
        self.log('val_mf1', avg_mf1, prog_bar=True)
        self.log('val_iou', avg_iou, prog_bar=True)
        
        # 可视化
        self._visualize_outputs(self.val_step_outputs, "验证")
        
        # 清空记录
        self.val_step_outputs = []
    
    def _visualize_outputs(self, outputs: List[Dict], prefix: str):
        """可视化输出结果"""
        if not outputs or outputs[0]['image'] is None:
            return
        
        # 获取第一个batch的数据
        image = outputs[0]['image']
        hr_feats = outputs[0]['outputs']['hr_feats']
        preds = outputs[0]['outputs']['logits'].argmax(dim=1)
        targets = outputs[0]['batch']['mask'].squeeze(1)
        
        # 创建可视化图像
        vis_images = self._create_visualization_grid(image, hr_feats, preds, targets, prefix)
        
        if hasattr(self.logger, "experiment") and hasattr(self.logger.experiment, "log"):
            self.logger.experiment.log({
                f"{prefix}_visualization": wandb.Image(vis_images)
            })
    
    def _create_visualization_grid(self, images, hr_feats, preds, targets, prefix):
        """创建一行四列的可视化网格"""
        # 确保数据在CPU上
        images = images.detach().cpu()
        hr_feats = hr_feats.detach().cpu()
        preds = preds.detach().cpu()
        targets = targets.detach().cpu()
        
        # 获取批次大小
        batch_size = min(images.shape[0], 8)  # 最多显示8张图
        
        # 创建特征可视化
        hr_feats_vis = self._visualize_features(hr_feats)
        
        # 创建预测掩码和目标掩码的可视化
        preds_vis = self._create_mask_visualization(preds, images)
        targets_vis = self._create_mask_visualization(targets, images)
        
        # 合并图像
        combined_images = []
        for i in range(batch_size):
            # 获取单个样本的图像
            orig_img = T.ToPILImage()(images[i])
            feat_img = hr_feats_vis[i]
            pred_img = preds_vis[i]
            target_img = targets_vis[i]
            
            # 确保所有图像大小相同
            width, height = orig_img.size
            feat_img = feat_img.resize((width, height))
            pred_img = pred_img.resize((width, height))
            target_img = target_img.resize((width, height))
            
            # 创建合并图像
            margin = 10
            total_width = width * 4 + margin * 5
            total_height = height + margin * 2
            
            combined = Image.new('RGB', (total_width, total_height), color=(255, 255, 255))
            
            # 粘贴图像
            combined.paste(orig_img, (margin, margin))
            combined.paste(feat_img, (margin * 2 + width, margin))
            combined.paste(pred_img, (margin * 3 + width * 2, margin))
            combined.paste(target_img, (margin * 4 + width * 3, margin))
            
            # 添加标题
            draw = ImageDraw.Draw(combined)
            try:
                font = ImageFont.truetype("Arial.ttf", 20)
            except IOError:
                font = ImageFont.load_default()
            
            titles = ["Image", "HR Feature", "Prediction", "Mask"]
            for j, title in enumerate(titles):
                x = margin * (j + 1) + width * j + width//2 - 40
                draw.text((x, 5), title, fill=(0, 0, 0), font=font)
            
            combined_images.append(combined)
        
        return combined_images[0] if combined_images else None
    
    def _visualize_features(self, features: torch.Tensor) -> List[Image.Image]:
        """可视化特征图"""
        batch_size = features.shape[0]
        vis_images = []
        
        for i in range(batch_size):
            # 将特征归一化到 [0, 1] 范围
            C, H, W = features[i].shape
            features_norm = features[i].clone()
            
            # 对每个通道分别归一化
            for c in range(C):
                channel = features[i, c]
                min_val = channel.min()
                max_val = channel.max()
                if max_val > min_val:
                    features_norm[c] = (channel - min_val) / (max_val - min_val)
            
            # 如果通道数大于3，使用PCA降维
            if C > 3:
                features_flat = features_norm.view(C, -1).transpose(0, 1)
                U, S, V = torch.pca_lowrank(features_flat, q=3)
                pca_features = torch.matmul(features_flat, V[:, :3]).view(H, W, 3)
                pca_features = pca_features.permute(2, 0, 1)
                
                # 归一化PCA结果
                for c in range(3):
                    channel = pca_features[c]
                    min_val = channel.min()
                    max_val = channel.max()
                    if max_val > min_val:
                        pca_features[c] = (channel - min_val) / (max_val - min_val)
                
                result = pca_features
            else:
                # 如果通道数小于等于3，直接使用前3个通道或填充
                result = torch.zeros(3, H, W)
                for c in range(min(C, 3)):
                    result[c] = features_norm[c]
            
            # 转换为PIL图像
            result = result.numpy().transpose(1, 2, 0)
            result = (result * 255).astype(np.uint8)
            result = Image.fromarray(result)
            vis_images.append(result)
        
        return vis_images
    
    def _create_mask_visualization(self, masks: torch.Tensor, original_images: torch.Tensor) -> List[Image.Image]:
        """创建掩码可视化"""
        batch_size = masks.shape[0]
        vis_images = []
        
        for i in range(batch_size):
            mask = masks[i].cpu().numpy()
            image = original_images[i].cpu().permute(1, 2, 0).numpy()
            
            # 归一化图像
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = np.clip(image, 0, 255).astype(np.uint8)
            
            # 创建彩色掩码
            height, width = mask.shape
            color_mask = np.zeros((height, width, 3), dtype=np.uint8)
            
            # 为每个类别分配不同的颜色
            unique_classes = np.unique(mask)
            for cls in unique_classes:
                if cls == 0:  # 背景类
                    continue
                color = np.random.randint(0, 255, 3, dtype=np.uint8)
                color_mask[mask == cls] = color
            
            # 创建半透明叠加效果
            alpha = 0.6
            overlay = (1 - alpha) * image + alpha * color_mask
            overlay = np.clip(overlay, 0, 255).astype(np.uint8)
            
            # 转换为PIL图像
            vis_images.append(Image.fromarray(overlay))
        
        return vis_images
    
    def get_trainable_params(self) -> List[nn.Parameter]:
        """获取可训练参数"""
        return [p for p in self.parameters() if p.requires_grad]
    
    def configure_optimizers(self):
        """配置优化器"""
        if self.optimizer is not None:
            # 如果优化器已经配置，直接使用它
            optimizer = self.optimizer(self.get_trainable_params())
            return optimizer
        
        # 默认使用AdamW优化器
        optimizer = torch.optim.AdamW(
            self.get_trainable_params(),
            lr=1e-4,
            weight_decay=0.0001
        )
        
        # 默认使用余弦退火学习率调度器
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "name": "lr"
            }
        }