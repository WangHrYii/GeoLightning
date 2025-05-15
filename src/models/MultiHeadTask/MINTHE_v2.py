import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from lightning import LightningModule
import rootutils
import numpy as np
import cv2
import math

rootutils.setup_root('/home/whr/Codes/GeoLightning/src/train.py', indicator=".project-root", pythonpath=True)


from src.models.backbones import DINOv2
from src.models.RegressionTask.dpt import DPTHead
from src.models.RegressionTask.dpt import _make_scratch, _make_fusion_block
from models.MultiHeadTask.TreeHeightBase import TreeHeightBase, ScaleInvariantLoss, GradientConsistencyLoss
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class HeadInteractionModule(nn.Module):
    """
    高级解码器头交互模块（Advanced Head Interaction Module）
    实现语义分割解码器与高度回归解码器之间的双向特征交互
    
    使用注意力机制和空间上下文信息，考虑了树木分割和高度回归任务的相关性
    """
    def __init__(self, features):
        super().__init__()
        self.features = features
        
        # 通道减少以降低计算复杂度（用于注意力计算前）
        self.seg_channel_reduce = nn.Conv2d(features, features // 2, kernel_size=1)
        self.height_channel_reduce = nn.Conv2d(features, features // 2, kernel_size=1)
        
        # 相互注意力 - 用于跨任务特征增强
        # 关键点/查询/值投影
        self.seg_to_height_key = nn.Conv2d(features // 2, features // 2, kernel_size=1)
        self.seg_to_height_query = nn.Conv2d(features // 2, features // 2, kernel_size=1)
        self.seg_to_height_value = nn.Conv2d(features // 2, features // 2, kernel_size=1)
        
        self.height_to_seg_key = nn.Conv2d(features // 2, features // 2, kernel_size=1)
        self.height_to_seg_query = nn.Conv2d(features // 2, features // 2, kernel_size=1)
        self.height_to_seg_value = nn.Conv2d(features // 2, features // 2, kernel_size=1)
        
        # 空间上下文编码 - 捕获每个任务内的空间关系
        self.seg_context = nn.Sequential(
            nn.Conv2d(features // 2, features // 2, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm2d(features // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(features // 2, features // 2, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(features // 2),
            nn.ReLU(inplace=True)
        )
        
        self.height_context = nn.Sequential(
            nn.Conv2d(features // 2, features // 2, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm2d(features // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(features // 2, features // 2, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(features // 2),
            nn.ReLU(inplace=True)
        )
        
        # 门控注意力机制 - 根据分割掩码控制高度特征的流动
        self.seg_gate = nn.Sequential(
            nn.Conv2d(features // 2, features // 2, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 最终融合模块
        self.seg_fusion = nn.Sequential(
            nn.Conv2d(features + features // 2, features, kernel_size=3, padding=1),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True)
        )
        
        self.height_fusion = nn.Sequential(
            nn.Conv2d(features + features // 2, features, kernel_size=3, padding=1),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True)
        )
        
        # 任务特定的后处理
        self.seg_postprocess = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True)
        )
        
        self.height_postprocess = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True)
        )
    
    def _cross_attention(self, query, key, value):
        """计算跨任务注意力
        
        Args:
            query: 查询特征 [B, C, H, W]
            key: 键特征 [B, C, H, W]
            value: 值特征 [B, C, H, W]
            
        Returns:
            attended_features: 注意力增强的特征 [B, C, H, W]
        """
        batch, channels, height, width = query.size()
        
        # 将特征reshape为注意力计算所需的形状
        query_flat = query.view(batch, channels, -1).permute(0, 2, 1)  # [B, HW, C]
        key_flat = key.view(batch, channels, -1)  # [B, C, HW]
        value_flat = value.view(batch, channels, -1).permute(0, 2, 1)  # [B, HW, C]
        
        # 计算注意力权重
        attention = torch.matmul(query_flat, key_flat)  # [B, HW, HW]
        attention = attention / (channels ** 0.5)  # 缩放因子
        attention = F.softmax(attention, dim=-1)
        
        # 应用注意力
        context = torch.matmul(attention, value_flat)  # [B, HW, C]
        context = context.permute(0, 2, 1).contiguous().view(batch, channels, height, width)
        
        return context
    
    def forward(self, seg_features, height_features):
        """
        解码器头交互
        
        Args:
            seg_features: 分割解码器特征, [B, C, H, W]
            height_features: 高度解码器特征, [B, C, H, W]
            
        Returns:
            enhanced_seg_features: 增强后的分割解码器特征
            enhanced_height_features: 增强后的高度解码器特征
        """
        batch, _, height, width = seg_features.size()
        
        # 保存原始特征用于残差连接
        seg_orig = seg_features
        height_orig = height_features
        
        # 降低通道数以减少计算复杂度
        seg_reduced = self.seg_channel_reduce(seg_features)
        height_reduced = self.height_channel_reduce(height_features)
        
        # 提取空间上下文信息
        seg_ctx = self.seg_context(seg_reduced)
        height_ctx = self.height_context(height_reduced)
        
        # 准备分割到高度的注意力组件
        s2h_query = self.height_to_seg_query(height_reduced)
        s2h_key = self.seg_to_height_key(seg_ctx)
        s2h_value = self.seg_to_height_value(seg_ctx)
        
        # 准备高度到分割的注意力组件
        h2s_query = self.seg_to_height_query(seg_reduced)
        h2s_key = self.height_to_seg_key(height_ctx)
        h2s_value = self.height_to_seg_value(height_ctx)
        
        # 计算跨任务注意力
        seg_attend_height = self._cross_attention(h2s_query, h2s_key, h2s_value)
        height_attend_seg = self._cross_attention(s2h_query, s2h_key, s2h_value)
        
        # 分割特征影响高度预测 - 通过门控机制
        # 树木分割的地方应该有更强的高度预测关注
        gate = self.seg_gate(seg_ctx)
        height_attend_seg = height_attend_seg * gate
        
        # 特征融合 - 将原始特征和注意力增强的特征连接起来
        enhanced_seg = torch.cat([seg_orig, seg_attend_height], dim=1)
        enhanced_height = torch.cat([height_orig, height_attend_seg], dim=1)
        
        # 最终融合
        enhanced_seg_features = self.seg_fusion(enhanced_seg)
        enhanced_height_features = self.height_fusion(enhanced_height)
        
        # 任务特定后处理
        enhanced_seg_features = self.seg_postprocess(enhanced_seg_features)
        enhanced_height_features = self.height_postprocess(enhanced_height_features)
        
        # 残差连接
        enhanced_seg_features = enhanced_seg_features + seg_orig
        enhanced_height_features = enhanced_height_features + height_orig
        
        return enhanced_seg_features, enhanced_height_features


class MINTHE_v2(nn.Module):
    """
    多任务交互式树木高度估计网络 v2 (Multitask Interactive Network for Tree Height Estimation v2)
    改进版本：解码器头之间直接交互而不是在DINOv2特征层面交互
    """
    def __init__(
        self,
        encoder='vitb',
        seg_features=128,
        height_features=128,
        out_channels=None,
        n_classes=1,
        use_bn=False,
        use_clstoken=False,
        bilinear=True,
        pretrained_weights=None,
        auto_adjust_channels=True
    ):
        super(MINTHE_v2, self).__init__()
        
        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11], 
            'vitl': [4, 11, 17, 23], 
            'vitg': [9, 19, 29, 39]
        }
        
        # 每个模型对应的特征通道数
        embed_dims = {
            'vits': 384,
            'vitb': 768,
            'vitl': 1024,
            'vitg': 1536
        }
        
        # 默认的Depth Anything v2通道配置 - 根据预训练权重中的配置
        depth_anything_channels = {
            'vits': [96, 192, 384, 768],
            'vitb': [96, 192, 384, 768],
            'vitl': [128, 256, 512, 1024],
            'vitg': [128, 256, 512, 1024]
        }
        
        # 如果需要使用预训练权重，并且设置了自动调整通道
        if pretrained_weights and auto_adjust_channels:
            # 预先读取权重文件以确定正确的通道配置
            try:
                import os
                import torch
                
                if not os.path.exists(pretrained_weights):
                    log.warning(f"警告: 找不到预训练权重文件 {pretrained_weights}, 将使用默认通道配置")
                    if out_channels is None:
                        self.out_channels = depth_anything_channels[encoder]
                else:
                    # 加载权重文件
                    checkpoint = torch.load(pretrained_weights, map_location='cpu')
                    if 'model' in checkpoint:
                        state_dict = checkpoint['model']
                    else:
                        state_dict = checkpoint
                    
                    # 检测原始预训练模型的out_channels
                    detected_channels = []
                    for i in range(4):
                        key = f'depth_head.projects.{i}.weight'
                        if key in state_dict:
                            # 权重形状为 [out_channel, in_channel, 1, 1]
                            detected_channels.append(state_dict[key].shape[0])
                        else:
                            # 如果找不到权重，使用默认值
                            detected_channels.append(depth_anything_channels[encoder][i])
                    
                    # 使用检测到的通道配置
                    self.out_channels = detected_channels
                    log.info(f"自动检测到预训练模型的out_channels配置: {self.out_channels}")
            except Exception as e:
                log.error(f"读取预训练权重文件时出错: {str(e)}")
                log.info("将使用默认通道配置")
                if out_channels is None:
                    self.out_channels = depth_anything_channels[encoder]
        # 根据模型确定输出通道数
        elif out_channels is None:
            if pretrained_weights:  # 如果指定了预训练权重，使用默认的Depth Anything通道配置
                self.out_channels = depth_anything_channels[encoder]
                log.info(f"使用预训练Depth Anything兼容的out_channels配置: {self.out_channels}")
            else:  # 否则使用一致的通道数
                embed_dim = embed_dims[encoder]
                self.out_channels = [embed_dim, embed_dim, embed_dim, embed_dim]    
                log.info(f"使用一致的out_channels配置: {self.out_channels}")
        else:
            self.out_channels = out_channels
            log.info(f"使用自定义out_channels配置: {self.out_channels}")

        # 共享编码器 (DINO V2)
        self.encoder = encoder
        self.pretrained = DINOv2(model_name=encoder)
        
        # 语义分割解码器 (基于DPT)
        from src.models.MultiHeadTask.MINTHE import SegmentationDPTHead
        self.seg_decoder = SegmentationDPTHead(
            in_channels=self.pretrained.embed_dim,
            features=seg_features, 
            use_bn=use_bn,
            out_channels=self.out_channels, 
            n_classes=n_classes, 
            use_clstoken=use_clstoken
        )
        
        # 高度回归解码器 (Depth Anything)
        self.height_head = DPTHead(
            self.pretrained.embed_dim, 
            height_features, 
            use_bn, 
            out_channels=self.out_channels, 
            use_clstoken=use_clstoken
        )
        
        # 解码器头交互模块 - 在refinenet的每个阶段添加交互
        # 这里用4个交互模块分别对refinenet的4个阶段进行交互
        self.interaction_modules = nn.ModuleList([
            HeadInteractionModule(seg_features)
            for _ in range(4)
        ])
        
        # 如果提供了预训练权重，加载它们
        if pretrained_weights:
            from src.models.MultiHeadTask.MINTHE import MINTHE
            # 创建一个临时MINTHE模型来加载权重
            temp_model = MINTHE(
                encoder=encoder, 
                seg_features=seg_features, 
                height_features=height_features,
                out_channels=self.out_channels,
                n_classes=n_classes,
                use_bn=use_bn,
                use_clstoken=use_clstoken,
                pretrained_weights=pretrained_weights,
                auto_adjust_channels=auto_adjust_channels
            )
            
            # 从临时模型复制权重
            self.pretrained.load_state_dict(temp_model.pretrained.state_dict())
            self.seg_decoder.load_state_dict(temp_model.seg_decoder.state_dict())
            self.height_head.load_state_dict(temp_model.height_head.state_dict())
            
            del temp_model
    
    def extract_features(self, x):
        """提取多尺度特征"""
        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14
        
        # 获取中间特征
        features = self.pretrained.get_intermediate_layers(
            x, 
            self.intermediate_layer_idx[self.encoder], 
            return_class_token=True
        )
        
        # 处理特征
        processed_features = []
        for feature in features:
            processed_features.append(feature)
            
        return processed_features, patch_h, patch_w
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入图像，形状为 [B, 3, H, W]
            
        Returns:
            seg_out: 语义分割结果
            height_out: 高度回归结果
        """
        # 1. 提取多尺度特征
        features, patch_h, patch_w = self.extract_features(x)
        
        # 2. 为分割和高度回归准备特征
        seg_features = features.copy()
        height_features = features.copy()
        
        # 3. 将特征传入各自的解码器
        # 这里修改forward实现以获取内部refinenet特征
        
        # 3.1 分割解码器中的特征提取部分
        # 从SegmentationDPTHead提取特征和层处理
        seg_layers = []
        for i, feature in enumerate(seg_features):
            feat = feature[0]
            feat = feat.permute(0, 2, 1).reshape((feat.shape[0], feat.shape[-1], patch_h, patch_w))
            
            feat = self.seg_decoder.projects[i](feat)
            feat = self.seg_decoder.resize_layers[i](feat)
            
            seg_layers.append(feat)
        
        # 3.2 高度解码器中的特征提取部分
        height_layers = []
        for i, feature in enumerate(height_features):
            feat = feature[0]
            feat = feat.permute(0, 2, 1).reshape((feat.shape[0], feat.shape[-1], patch_h, patch_w))
            
            feat = self.height_head.projects[i](feat)
            feat = self.height_head.resize_layers[i](feat)
            
            height_layers.append(feat)
        
        # 3.3 特征转换为refinenet所需的输入
        layer_1_s, layer_2_s, layer_3_s, layer_4_s = seg_layers
        layer_1_h, layer_2_h, layer_3_h, layer_4_h = height_layers
        
        # 转换为refinenet特征
        layer_1_s_rn = self.seg_decoder.scratch.layer1_rn(layer_1_s)
        layer_2_s_rn = self.seg_decoder.scratch.layer2_rn(layer_2_s)
        layer_3_s_rn = self.seg_decoder.scratch.layer3_rn(layer_3_s)
        layer_4_s_rn = self.seg_decoder.scratch.layer4_rn(layer_4_s)
        
        layer_1_h_rn = self.height_head.scratch.layer1_rn(layer_1_h)
        layer_2_h_rn = self.height_head.scratch.layer2_rn(layer_2_h)
        layer_3_h_rn = self.height_head.scratch.layer3_rn(layer_3_h)
        layer_4_h_rn = self.height_head.scratch.layer4_rn(layer_4_h)
        
        # 4. Refinenet 路径上的特征交互
        # 4.1 Refinenet 4 层交互
        path_4_s = self.seg_decoder.scratch.refinenet4(layer_4_s_rn, size=layer_3_s_rn.shape[2:]) 
        path_4_h = self.height_head.scratch.refinenet4(layer_4_h_rn, size=layer_3_h_rn.shape[2:])
        
        # 交互
        path_4_s, path_4_h = self.interaction_modules[0](path_4_s, path_4_h)
        
        # 4.2 Refinenet 3 层交互
        path_3_s = self.seg_decoder.scratch.refinenet3(path_4_s, layer_3_s_rn, size=layer_2_s_rn.shape[2:])
        path_3_h = self.height_head.scratch.refinenet3(path_4_h, layer_3_h_rn, size=layer_2_h_rn.shape[2:])
        
        # 交互
        path_3_s, path_3_h = self.interaction_modules[1](path_3_s, path_3_h)
        
        # 4.3 Refinenet 2 层交互
        path_2_s = self.seg_decoder.scratch.refinenet2(path_3_s, layer_2_s_rn, size=layer_1_s_rn.shape[2:])
        path_2_h = self.height_head.scratch.refinenet2(path_3_h, layer_2_h_rn, size=layer_1_h_rn.shape[2:])
        
        # 交互
        path_2_s, path_2_h = self.interaction_modules[2](path_2_s, path_2_h)
        
        # 4.4 Refinenet 1 层交互
        path_1_s = self.seg_decoder.scratch.refinenet1(path_2_s, layer_1_s_rn)
        path_1_h = self.height_head.scratch.refinenet1(path_2_h, layer_1_h_rn)
        
        # 交互
        path_1_s, path_1_h = self.interaction_modules[3](path_1_s, path_1_h)
        
        # 5. 输出层处理
        seg_out = self.seg_decoder.scratch.output_conv1(path_1_s)
        seg_out = F.interpolate(seg_out, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True)
        seg_out = self.seg_decoder.scratch.output_conv2(seg_out)
        
        height_out = self.height_head.scratch.output_conv1(path_1_h)
        height_out = F.interpolate(height_out, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True)
        height_out = self.height_head.scratch.output_conv2(height_out)
        
        return seg_out, height_out
    
    @torch.no_grad()
    def infer_image(self, raw_image, input_size=518):
        """
        对单张图像进行推理
        
        Args:
            raw_image: 原始图像，BGR格式的numpy数组
            input_size: 输入大小
            
        Returns:
            seg_mask: 语义分割掩码
            height_map: 高度图
        """
        # 图像预处理
        from src.models.RegressionTask.dpt import Resize, NormalizeImage, PrepareForNet
        import cv2
        
        transform = nn.Sequential(
            Resize(
                width=input_size,
                height=input_size,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        )
        
        h, w = raw_image.shape[:2]
        
        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
        
        image = transform({'image': image})['image']
        image = torch.from_numpy(image).unsqueeze(0)
        
        DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        image = image.to(DEVICE)
        
        # 前向传播
        seg_out, height_out = self.forward(image)
        
        # 后处理
        seg_mask = F.interpolate(seg_out, (h, w), mode="bilinear", align_corners=True)
        seg_mask = torch.sigmoid(seg_mask)[0, 0].cpu().numpy()
        
        height_map = F.interpolate(height_out.unsqueeze(1), (h, w), mode="bilinear", align_corners=True)
        height_map = height_map[0, 0].cpu().numpy()
        
        return seg_mask, height_map


class MINTHE_v2Module(TreeHeightBase):
    """
    MINTHE_v2的Lightning模块实现
    """
    def __init__(
        self,
        encoder='vitl',
        seg_features=128,
        height_features=128,
        out_channels=None,
        n_classes=1,
        use_bn=False,
        use_clstoken=False,
        bilinear=True,
        learning_rate=1e-4,
        weight_decay=1e-5,
        lambda_seg=1.0,
        lambda_height=1.0,
        si_lambda=0.5,
        gradient_lambda=0.1,
        lr_scheduler_patience=5,
        lr_scheduler_factor=0.5,
        pretrained_depth_weights='/home/whr/Codes/GeoLightning/ckpts/depth_anything_v2_vitb.pth',
        auto_adjust_channels=True,
        backbone_lr=1e-5,
        seg_head_lr=1e-4,
        interaction_modules_lr=1e-4,
        height_head_lr=1e-4,
        precision='16-mixed'
    ):
        # 首先调用父类的__init__方法，传递一个None作为model参数
        # 在创建完MINTHE_v2模型后，我们会手动设置self.model
        super().__init__(
            model=None,
            n_classes=n_classes,
            lambda_seg=lambda_seg,
            lambda_height=lambda_height,
            si_lambda=si_lambda,
            gradient_lambda=gradient_lambda
        )
        
        # Save all hyperparameters
        self.save_hyperparameters()
        
        # Create MINTHE_v2 model
        self.minthe_v2_model = MINTHE_v2(
            encoder=encoder,
            seg_features=seg_features,
            height_features=height_features,
            out_channels=out_channels,
            n_classes=n_classes,
            use_bn=use_bn,
            use_clstoken=use_clstoken,
            bilinear=bilinear,
            pretrained_weights=pretrained_depth_weights,
            auto_adjust_channels=auto_adjust_channels
        )
        
        # 手动设置model属性，使其指向minthe_v2_model
        self.model = self.minthe_v2_model
    
    def training_step(self, batch, batch_idx):
        """Override training_step to log learning rates"""
        # Log current learning rates
        for i, param_group in enumerate(self.trainer.optimizers[0].param_groups):
            if i == 0:
                self.log("lr/backbone", param_group['lr'], on_step=True)
            elif i == 1:
                self.log("lr/seg_head", param_group['lr'], on_step=True)
            elif i == 2:
                self.log("lr/interaction", param_group['lr'], on_step=True)
            elif i == 3:
                self.log("lr/height_head", param_group['lr'], on_step=True)

        # Call the parent training_step
        total_loss = super().training_step(batch, batch_idx)
        
        return total_loss
    
    def configure_optimizers(self):
        """Configure optimizers and learning rate scheduler"""
        # Define parameter groups with different learning rates
        param_groups = [
            {
                'params': self.minthe_v2_model.pretrained.parameters(),
                'lr': self.hparams.backbone_lr,
                'weight_decay': self.hparams.weight_decay
            },
            {
                'params': self.minthe_v2_model.seg_decoder.parameters(),
                'lr': self.hparams.seg_head_lr,
                'weight_decay': self.hparams.weight_decay
            },
            {
                'params': self.minthe_v2_model.interaction_modules.parameters(),
                'lr': self.hparams.interaction_modules_lr,
                'weight_decay': self.hparams.weight_decay
            },
            {
                'params': self.minthe_v2_model.height_head.parameters(),
                'lr': self.hparams.height_head_lr,
                'weight_decay': self.hparams.weight_decay
            }
        ]
        
        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=self.hparams.weight_decay
        )
        
        # Configure learning rate scheduler with warmup and cosine decay
        num_training_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(num_training_steps * 0.1)  # 10% warmup
        
        def lr_lambda(step):
            if step < warmup_steps:
                # Linear warmup
                return float(step) / float(max(1, warmup_steps))
            else:
                # Cosine decay
                progress = float(step - warmup_steps) / float(max(1, num_training_steps - warmup_steps))
                return 0.5 * (1.0 + math.cos(math.pi * progress))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
