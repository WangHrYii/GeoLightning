import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from lightning import LightningModule
import rootutils
import numpy as np
import cv2
import math

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.backbones import DINOv2
from src.models.RegressionTask.dpt import DPTHead
from src.models.RegressionTask.dpt import _make_scratch, _make_fusion_block
from src.models.MultiHeadTask.TreeHeightBase import TreeHeightBase
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class BidirectionalFeatureFusionInteraction(nn.Module):
    """
    双向特征融合交互模块（Bidirectional Feature Fusion Interaction, BFFI）
    实现语义分割任务与高度回归任务之间的有效信息交换
    """
    def __init__(self, features):
        super().__init__()
        self.features = features
        
        # 分割引导的高度注意力
        self.seg_query = nn.Conv2d(features, features // 4, kernel_size=1)
        self.height_key = nn.Conv2d(features, features // 4, kernel_size=1)
        self.height_value = nn.Conv2d(features, features, kernel_size=1)
        
        # 高度引导的分割注意力
        self.height_query = nn.Conv2d(features, features // 4, kernel_size=1)
        self.seg_key = nn.Conv2d(features, features // 4, kernel_size=1)
        self.seg_value = nn.Conv2d(features, features, kernel_size=1)
        
        # 特征融合层
        self.height_fusion = nn.Sequential(
            nn.Conv2d(features * 2, features, kernel_size=3, padding=1),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True)
        )
        
        self.seg_fusion = nn.Sequential(
            nn.Conv2d(features * 2, features, kernel_size=3, padding=1),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, seg_features, height_features):
        """
        双向特征融合交互
        
        Args:
            seg_features: 语义分割特征，形状为 [B, C, H, W]
            height_features: 高度回归特征，形状为 [B, C, H, W]
            
        Returns:
            enhanced_seg_features: 增强后的语义分割特征
            enhanced_height_features: 增强后的高度回归特征
        """
        batch_size, _, h, w = seg_features.shape
        
        # 1. 分割引导的高度注意力
        # 使用分割特征作为query，高度特征作为key和value
        query = self.seg_query(seg_features).view(batch_size, -1, h * w).permute(0, 2, 1)  # B, HW, C/4
        key = self.height_key(height_features).view(batch_size, -1, h * w)  # B, C/4, HW
        value = self.height_value(height_features).view(batch_size, -1, h * w).permute(0, 2, 1)  # B, HW, C
        
        # 计算注意力权重
        attention_weights = torch.bmm(query, key) / (self.features ** 0.5)  # B, HW, HW
        attention_weights = F.softmax(attention_weights, dim=-1)
        
        # 应用注意力
        height_context = torch.bmm(attention_weights, value)  # B, HW, C
        height_context = height_context.permute(0, 2, 1).view(batch_size, -1, h, w)  # B, C, H, W
        
        # 2. 高度引导的分割注意力
        # 使用高度特征作为query，分割特征作为key和value
        query = self.height_query(height_features).view(batch_size, -1, h * w).permute(0, 2, 1)  # B, HW, C/4
        key = self.seg_key(seg_features).view(batch_size, -1, h * w)  # B, C/4, HW
        value = self.seg_value(seg_features).view(batch_size, -1, h * w).permute(0, 2, 1)  # B, HW, C
        
        # 计算注意力权重
        attention_weights = torch.bmm(query, key) / (self.features ** 0.5)  # B, HW, HW
        attention_weights = F.softmax(attention_weights, dim=-1)
        
        # 应用注意力
        seg_context = torch.bmm(attention_weights, value)  # B, HW, C
        seg_context = seg_context.permute(0, 2, 1).view(batch_size, -1, h, w)  # B, C, H, W
        
        # 3. 特征融合
        # 将原始特征和注意力特征拼接后融合
        enhanced_height_features = self.height_fusion(torch.cat([height_features, height_context], dim=1))
        enhanced_seg_features = self.seg_fusion(torch.cat([seg_features, seg_context], dim=1))
        
        return enhanced_seg_features, enhanced_height_features


class SegmentationDPTHead(nn.Module):
    """
    基于DPT的语义分割解码器，结构与DPTHead相似
    """
    def __init__(
        self, 
        in_channels, 
        features=64, 
        use_bn=False, 
        out_channels=[768, 768, 768, 768], 
        n_classes=1,
        use_clstoken=False
    ):
        super(SegmentationDPTHead, self).__init__()
        
        self.use_clstoken = use_clstoken
        self.n_classes = n_classes
        
        # 特征投影层
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])
        
        # 特征尺寸调整层
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])
        
        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2 * in_channels, in_channels),
                        nn.GELU()))
        
        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )
        
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)
        
        # 输出卷积层
        head_features_1 = features
        head_features_2 = 32
        
        self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)
        
        # 分割输出层
        if n_classes == 1:
            self.scratch.output_conv2 = nn.Sequential(
                nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(head_features_2, n_classes, kernel_size=1, stride=1, padding=0),
            )
        else:
            self.scratch.output_conv2 = nn.Sequential(
                nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(head_features_2, n_classes, kernel_size=1, stride=1, padding=0),
            )
    
    def forward(self, out_features, patch_h, patch_w):
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]
            
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            
            out.append(x)
        
        layer_1, layer_2, layer_3, layer_4 = out
        
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])        
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        
        out = self.scratch.output_conv1(path_1)
        out = F.interpolate(out, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True)
        out = self.scratch.output_conv2(out)

        return out


class MINTHE(nn.Module):
    """
    多任务交互式树木高度估计网络 (Multitask Interactive Network for Tree Height Estimation)
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
        super(MINTHE, self).__init__()
        
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
                    # print(f"自动检测到预训练模型的out_channels配置: {self.out_channels}")
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
                log.info(f"使用预训练Depth Anything兼容的out_channels配置: {self.out_channels}")
        else:
            self.out_channels = out_channels
            # print(f"使用自定义out_channels配置: {self.out_channels}")
            log.info(f"使用自定义out_channels配置: {self.out_channels}")

        # 共享编码器 (DINO V2)
        self.encoder = encoder
        self.pretrained = DINOv2(model_name=encoder)
        
        # 语义分割解码器 (基于DPT)
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
        
        # 双向特征融合交互模块
        self.interaction_modules = nn.ModuleList([
            BidirectionalFeatureFusionInteraction(self.pretrained.embed_dim)
            for i in range(len(self.out_channels))
        ])
        
        # 如果提供了预训练权重，加载它们
        if pretrained_weights:
            self.load_depth_anything_weights(pretrained_weights)
    
    def load_depth_anything_weights(self, checkpoint_path, strict=False):
        """
        加载Depth Anything v2的预训练权重到backbone和高度估计头部
        
        Args:
            checkpoint_path: Depth Anything v2的权重文件路径
            strict: 是否严格匹配参数名称
            
        Returns:
            成功加载的信息
        """
        # 导入必要的模块
        from src.models.RegressionTask.dpt import DepthAnythingV2
        import os
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"找不到权重文件: {checkpoint_path}")
        
        print(f"正在从 {checkpoint_path} 加载Depth Anything v2权重...")
        
        try:
            # 确定当前模型是否使用batch norm
            # 检查refinenet中的ResidualConvUnit是否有bn属性，而且它的值是True
            # ResidualConvUnit在初始化时会设置self.bn = bn
            use_bn = False
            if hasattr(self.height_head.scratch.refinenet1.resConfUnit1, 'bn'):
                use_bn = self.height_head.scratch.refinenet1.resConfUnit1.bn
            
            # 尝试加载检查点以获取其原始配置
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            
            # 检测原始预训练模型的out_channels
            # 这是基于对权重形状的分析得出的预训练模型配置
            pretrained_out_channels = []
            for i in range(4):
                key = f'depth_head.projects.{i}.weight'
                if key in state_dict:
                    # 权重形状为 [out_channel, in_channel, 1, 1]
                    pretrained_out_channels.append(state_dict[key].shape[0])
                else:
                    # 如果找不到权重，使用默认值
                    if i == 0:
                        pretrained_out_channels.append(96)  # ViTS
                    elif i == 1:
                        pretrained_out_channels.append(192)
                    elif i == 2:
                        pretrained_out_channels.append(384)
                    else:
                        pretrained_out_channels.append(768)
            
            # 1. 创建与预训练模型匹配的临时Depth Anything模型
            depth_model = DepthAnythingV2(
                encoder=self.encoder,
                features=self.height_head.scratch.layer1_rn.out_channels,  # 获取与当前模型匹配的特征数
                out_channels=pretrained_out_channels,  # 使用预训练模型的通道配置
                use_bn=use_bn,
                use_clstoken=self.height_head.use_clstoken
            )
            
            # 2. 加载权重到临时模型
            # 移除可能的module.前缀（如果是使用DDP训练的权重）
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}
            
            # 加载临时模型
            missing_keys, unexpected_keys = depth_model.load_state_dict(state_dict, strict=strict)
            if len(missing_keys) > 0:
                log.error(f"加载深度模型时缺少的键: {missing_keys}")
            if len(unexpected_keys) > 0:
                log.error(f"加载深度模型时未预期的键: {unexpected_keys}")
            
            # 3. 从临时模型复制backbone权重 (不受out_channels影响)
            encoder_dict = {
                k: v for k, v in depth_model.pretrained.state_dict().items()
            }
            missing_keys_encoder, unexpected_keys_encoder = self.pretrained.load_state_dict(encoder_dict, strict=strict)
            
            # # 4. 加载特定参数
            # print("加载特定参数...")
            
            # # 加载四个refinenet的权重 (这些与out_channels无关)
            # for i in range(1, 5):
            #     refinenet_dict = depth_model.depth_head.scratch.__getattr__(f'refinenet{i}').state_dict()
            #     self.height_head.scratch.__getattr__(f'refinenet{i}').load_state_dict(refinenet_dict)
            #     print(f"成功加载refinenet{i}的权重")
            
            # # 加载output_conv的权重 (这些与out_channels无关)
            # output_conv1_dict = depth_model.depth_head.scratch.output_conv1.state_dict() 
            # self.height_head.scratch.output_conv1.load_state_dict(output_conv1_dict)
            # print("成功加载output_conv1的权重")
            
            # output_conv2_dict = depth_model.depth_head.scratch.output_conv2.state_dict()
            # self.height_head.scratch.output_conv2.load_state_dict(output_conv2_dict)
            # print("成功加载output_conv2的权重")
            
            # # 5. 尝试加载项目层权重
            # projects_loaded = 0
            # for i in range(4):
            #     # 检查当前模型与预训练模型的通道数是否兼容
            #     if self.out_channels[i] == pretrained_out_channels[i]:
            #         # 直接加载整个层
            #         proj_dict = depth_model.depth_head.projects[i].state_dict()
            #         self.height_head.projects[i].load_state_dict(proj_dict)
            #         projects_loaded += 1
            #         print(f"成功加载project{i}的完整权重")
            #     else:
            #         # 尝试部分加载权重（按最小维度截断）
            #         src_weight = depth_model.depth_head.projects[i].weight
            #         src_bias = depth_model.depth_head.projects[i].bias
                    
            #         # 获取目标权重和偏置的形状
            #         target_weight = self.height_head.projects[i].weight
            #         target_bias = self.height_head.projects[i].bias
                    
            #         # 确定共同的输出通道数
            #         common_out_channels = min(src_weight.size(0), target_weight.size(0))
                    
            #         # 复制共同部分的权重和偏置
            #         with torch.no_grad():
            #             self.height_head.projects[i].weight[:common_out_channels] = src_weight[:common_out_channels]
            #             self.height_head.projects[i].bias[:common_out_channels] = src_bias[:common_out_channels]
                    
            #         print(f"部分加载project{i}的权重 ({common_out_channels}/{self.out_channels[i]} 通道)")
                
            #     # 尝试加载resize层权重（如果out_channels匹配）
            #     if self.out_channels[i] == pretrained_out_channels[i]:
            #         try:
            #             resize_dict = depth_model.depth_head.resize_layers[i].state_dict()
            #             self.height_head.resize_layers[i].load_state_dict(resize_dict)
            #             print(f"成功加载resize_layer{i}的权重")
            #         except Exception as e:
            #             print(f"加载resize_layer{i}时出错: {str(e)}")
            
            # # 6. 尝试加载layer_n_rn权重
            # for n in range(1, 5):
            #     try:
            #         layer_dict = depth_model.depth_head.scratch.__getattr__(f'layer{n}_rn').state_dict()
            #         self.height_head.scratch.__getattr__(f'layer{n}_rn').load_state_dict(layer_dict)
            #         print(f"成功加载layer{n}_rn的权重")
            #     except Exception as e:
            #         print(f"无法加载layer{n}_rn权重: {str(e)}")
            
            # # 7. 返回加载信息
            # print(f"\n成功加载backbone和高度估计头部的权重！")
            # print(f"- 预训练模型的out_channels: {pretrained_out_channels}")
            # print(f"- 当前模型的out_channels: {self.out_channels}")
            # print(f"- 成功完整加载的project层: {projects_loaded}/4")
            # print(f"- Backbone: 缺少键 {len(missing_keys_encoder)}个, 未预期键 {len(unexpected_keys_encoder)}个")
            
            # 注意分割头部保持随机初始化
            log.info("注意：分割头部(SegmentationDPTHead)没有加载预训练权重，保持随机初始化")
            
            return {
                'encoder': {'missing': missing_keys_encoder, 'unexpected': unexpected_keys_encoder},
                'pretrained_out_channels': pretrained_out_channels,
                # 'projects_loaded': projects_loaded
            }
            
        except Exception as e:
            raise e
    
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
        for i, feature in enumerate(features):
            feat = feature[0]  # 不使用类别token
            # 将特征重塑为空间特征图 - 注意这里不再进行重塑，因为DPT头会处理
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
        # 提取多尺度特征
        features, patch_h, patch_w = self.extract_features(x)
        
        # 初始化分割和高度特征
        seg_features = features.copy()
        height_features = features.copy()
        
        # 应用双向特征融合交互 - 注意由于DPT头的输入格式不同，这里需要修改交互方式
        for i, interaction in enumerate(self.interaction_modules):
            # 将特征转换为适合交互模块的格式
            sf = seg_features[i][0].permute(0, 2, 1).reshape((seg_features[i][0].shape[0], seg_features[i][0].shape[-1], patch_h, patch_w))
            hf = height_features[i][0].permute(0, 2, 1).reshape((height_features[i][0].shape[0], height_features[i][0].shape[-1], patch_h, patch_w))
            
            # 应用交互
            sf_enhanced, hf_enhanced = interaction(sf, hf)
            
            # 将增强的特征转换回原始格式
            sf_token = sf_enhanced.flatten(2).permute(0, 2, 1)
            hf_token = hf_enhanced.flatten(2).permute(0, 2, 1)
            
            # 更新特征
            seg_features[i] = (sf_token, seg_features[i][1])
            height_features[i] = (hf_token, height_features[i][1])
        
        # 通过各自的解码器生成最终输出
        seg_out = self.seg_decoder(seg_features, patch_h, patch_w)
        height_out = self.height_head(height_features, patch_h, patch_w)
        
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


class MINTHEModule(TreeHeightBase):
    """
    MINTHE implementation based on TreeHeightBase
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
        pretrained_depth_weights=None,
        auto_adjust_channels=True,
        backbone_lr=1e-5,
        seg_head_lr=1e-4,
        interaction_modules_lr=1e-5,
        height_head_lr=1e-4,
        precision='16-mixed'
    ):
        # 首先调用父类的__init__方法，传递一个None作为model参数
        # 在创建完MINTHE模型后，我们会手动设置self.model
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
        
        # Create MINTHE model
        self.minthe_model = MINTHE(
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
        
        # 手动设置model属性，使其指向minthe_model
        self.model = self.minthe_model
    
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
                'params': self.minthe_model.pretrained.parameters(),
                'lr': self.hparams.backbone_lr,
                'weight_decay': self.hparams.weight_decay
            },
            {
                'params': self.minthe_model.seg_decoder.parameters(),
                'lr': self.hparams.seg_head_lr,
                'weight_decay': self.hparams.weight_decay
            },
            {
                'params': self.minthe_model.interaction_modules.parameters(),
                'lr': self.hparams.interaction_modules_lr,
                'weight_decay': self.hparams.weight_decay
            },
            {
                'params': self.minthe_model.height_head.parameters(),
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


if __name__ == "__main__":
    """
    用于测试计算过程是否存在错误的主函数
    """
    import cv2
    import matplotlib.pyplot as plt
    import os
    import time
    
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 创建模型
    encoder_type = 'vitb'  # 使用较小的模型进行快速测试
    model = MINTHE(encoder=encoder_type, pretrained_weights='ckpts/depth_anything_v2_vitb.pth')
    model = model.to(device)
    model.eval()
    
    # 打印模型配置
    print(f"Model encoder: {encoder_type}")
    print(f"Model embed_dim: {model.pretrained.embed_dim}")
    print(f"Model out_channels: {model.out_channels}")
    
    # 生成随机输入数据进行测试
    batch_size = 2
    image_size = 518  # 确保是14的倍数
    
    # 方法1：使用随机数据
    random_input = torch.randn(batch_size, 3, image_size, image_size).to(device)
    
    # 方法2：如果有测试图像，可以加载实际图像
    test_image_path = 'images/3.jpg'
    
    if test_image_path and os.path.exists(test_image_path):
        # 加载并预处理图像
        image = cv2.imread(test_image_path)
        if image is not None:
            image = cv2.resize(image, (image_size, image_size))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
            image = image.transpose(2, 0, 1)  # HWC to CHW
            image = torch.from_numpy(image).float().unsqueeze(0).to(device)
            
            # 复制到batch_size
            input_tensor = image.repeat(batch_size, 1, 1, 1)
        else:
            print(f"Failed to load image from {test_image_path}, using random input instead")
            input_tensor = random_input
    else:
        input_tensor = random_input
    
    try:
        # 测量推理时间
        start_time = time.time()
        
        # 前向传播
        with torch.no_grad():
            seg_out, height_out = model(input_tensor)
        
        end_time = time.time()
        
        # 输出形状和统计信息
        print(f"Input shape: {input_tensor.shape}")
        print(f"Segmentation output shape: {seg_out.shape}")
        print(f"Height output shape: {height_out.shape}")
        print(f"Inference time: {end_time - start_time:.4f} seconds")
        
        # 基本数值检查
        print(f"Segmentation output min: {seg_out.min().item()}, max: {seg_out.max().item()}")
        print(f"Height output min: {height_out.min().item()}, max: {height_out.max().item()}")
        
        # 测试各个模块
        print("\nTesting individual components:")
        
        # 测试特征提取
        features, patch_h, patch_w = model.extract_features(input_tensor)
        print(f"Patch dimensions: h={patch_h}, w={patch_w}")
        
        # 详细打印每个特征层的维度
        for i, feature in enumerate(features):
            print(f"Feature {i} shape: {feature[0].shape}")
            print(f"Feature {i} min: {feature[0].min().item():.4f}, max: {feature[0].max().item():.4f}, mean: {feature[0].mean().item():.4f}")
        
        # 测试分割解码器投影层
        print("\nTesting segmentation head projection layers:")
        projected_features = []
        for i, feature in enumerate(features):
            x = feature[0]
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            proj = model.seg_decoder.projects[i](x)
            resized = model.seg_decoder.resize_layers[i](proj)
            print(f"Feature {i} projection: {x.shape} -> {proj.shape} -> {resized.shape}")
            projected_features.append(resized)
            
        # 测试refinenet路径
        print("\nTesting refinenet path:")
        layer_1, layer_2, layer_3, layer_4 = projected_features
        
        layer_1_rn = model.seg_decoder.scratch.layer1_rn(layer_1)
        layer_2_rn = model.seg_decoder.scratch.layer2_rn(layer_2)
        layer_3_rn = model.seg_decoder.scratch.layer3_rn(layer_3)
        layer_4_rn = model.seg_decoder.scratch.layer4_rn(layer_4)
        
        print(f"layer_1_rn shape: {layer_1_rn.shape}")
        print(f"layer_2_rn shape: {layer_2_rn.shape}")
        print(f"layer_3_rn shape: {layer_3_rn.shape}")
        print(f"layer_4_rn shape: {layer_4_rn.shape}")
        
        # 测试交互模块
        seg_features = features.copy()
        height_features = features.copy()
        
        for i, interaction in enumerate(model.interaction_modules):
            print(f"Testing interaction module {i}")
            # 将特征转换为适合交互模块的格式
            sf = seg_features[i][0].permute(0, 2, 1).reshape((seg_features[i][0].shape[0], seg_features[i][0].shape[-1], patch_h, patch_w))
            hf = height_features[i][0].permute(0, 2, 1).reshape((height_features[i][0].shape[0], height_features[i][0].shape[-1], patch_h, patch_w))
            
            enhanced_seg, enhanced_height = interaction(sf, hf)
            print(f"  Enhanced segmentation feature shape: {enhanced_seg.shape}")
            print(f"  Enhanced height feature shape: {enhanced_height.shape}")
        
        # 测试分割解码器
        print("\nTesting segmentation decoder")
        seg_result = model.seg_decoder(features, patch_h, patch_w)
        print(f"Segmentation decoder output shape: {seg_result.shape}")
        
        # 测试高度解码器
        print("\nTesting height decoder")
        height_result = model.height_head(features, patch_h, patch_w)
        print(f"Height decoder output shape: {height_result.shape}")
        
        print("\nAll tests passed successfully!")
        
        # 可视化结果（可选）
        if batch_size > 0:
            # 处理分割结果
            if model.seg_decoder.n_classes == 1:
                seg_vis = torch.sigmoid(seg_out[0, 0]).cpu().numpy()
            else:
                seg_vis = torch.argmax(seg_out[0], dim=0).cpu().numpy()
            
            # 处理高度结果
            height_vis = height_out[0].cpu().numpy().squeeze()
            
            # 创建可视化
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 3, 1)
            plt.imshow(input_tensor[0].permute(1, 2, 0).cpu().numpy())
            plt.title("Input Image")
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            plt.imshow(seg_vis, cmap='gray')
            plt.title("Segmentation Output")
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            plt.imshow(height_vis, cmap='viridis') # 使用viridis颜色映射
            plt.title("Height Output")
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig("debug_output.png")
            print("Visualization saved as debug_output.png")
    
    except Exception as e:
        print(f"Error during test: {str(e)}")
        import traceback
        traceback.print_exc()



