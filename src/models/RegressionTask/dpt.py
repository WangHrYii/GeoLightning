"""
implementation of DepthAnythingV2
Author: Haoran Wang
"""
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose
from typing import Any, Dict, Optional, Tuple, Union, List, Callable
import torchmetrics

import rootutils

rootutils.setup_root('/home/whr/Codes/GeoLightning/src/train.py', indicator=".project-root", pythonpath=True)


from src.models.backbones import DINOv2
from src.models.RegressionTask.base import BaseRegressionTask


def _make_scratch(in_shape, out_shape, groups=1, expand=False):
    scratch = nn.Module()

    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    if len(in_shape) >= 4:
        out_shape4 = out_shape

    if expand:
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        if len(in_shape) >= 4:
            out_shape4 = out_shape * 8

    scratch.layer1_rn = nn.Conv2d(in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer2_rn = nn.Conv2d(in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer3_rn = nn.Conv2d(in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    if len(in_shape) >= 4:
        scratch.layer4_rn = nn.Conv2d(in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)

    return scratch


class ResidualConvUnit(nn.Module):
    """Residual convolution module.
    """

    def __init__(self, features, activation, bn):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.bn = bn

        self.groups=1

        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)
        
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)

        if self.bn == True:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.activation = activation

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """
        
        out = self.activation(x)
        out = self.conv1(out)
        if self.bn == True:
            out = self.bn1(out)
       
        out = self.activation(out)
        out = self.conv2(out)
        if self.bn == True:
            out = self.bn2(out)

        if self.groups > 1:
            out = self.conv_merge(out)

        return self.skip_add.add(out, x)


class FeatureFusionBlock(nn.Module):
    """Feature fusion block.
    """

    def __init__(
        self, 
        features, 
        activation, 
        deconv=False, 
        bn=False, 
        expand=False, 
        align_corners=True,
        size=None
    ):
        """Init.
        
        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners

        self.groups=1

        self.expand = expand
        out_features = features
        if self.expand == True:
            out_features = features // 2
        
        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)

        self.resConfUnit1 = ResidualConvUnit(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn)
        
        self.skip_add = nn.quantized.FloatFunctional()

        self.size=size

    def forward(self, *xs, size=None):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        output = self.resConfUnit2(output)

        if (size is None) and (self.size is None):
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}

        output = nn.functional.interpolate(output, **modifier, mode="bilinear", align_corners=self.align_corners)
        
        output = self.out_conv(output)

        return output
    
class Resize(object):
    """Resize sample to given size (width, height).
    """

    def __init__(
        self,
        width,
        height,
        resize_target=True,
        keep_aspect_ratio=False,
        ensure_multiple_of=1,
        resize_method="lower_bound",
        image_interpolation_method=cv2.INTER_AREA,
    ):
        """Init.

        Args:
            width (int): desired output width
            height (int): desired output height
            resize_target (bool, optional):
                True: Resize the full sample (image, mask, target).
                False: Resize image only.
                Defaults to True.
            keep_aspect_ratio (bool, optional):
                True: Keep the aspect ratio of the input sample.
                Output sample might not have the given width and height, and
                resize behaviour depends on the parameter 'resize_method'.
                Defaults to False.
            ensure_multiple_of (int, optional):
                Output width and height is constrained to be multiple of this parameter.
                Defaults to 1.
            resize_method (str, optional):
                "lower_bound": Output will be at least as large as the given size.
                "upper_bound": Output will be at max as large as the given size. (Output size might be smaller than given size.)
                "minimal": Scale as least as possible.  (Output size might be smaller than given size.)
                Defaults to "lower_bound".
        """
        self.__width = width
        self.__height = height

        self.__resize_target = resize_target
        self.__keep_aspect_ratio = keep_aspect_ratio
        self.__multiple_of = ensure_multiple_of
        self.__resize_method = resize_method
        self.__image_interpolation_method = image_interpolation_method

    def constrain_to_multiple_of(self, x, min_val=0, max_val=None):
        y = (np.round(x / self.__multiple_of) * self.__multiple_of).astype(int)

        if max_val is not None and y > max_val:
            y = (np.floor(x / self.__multiple_of) * self.__multiple_of).astype(int)

        if y < min_val:
            y = (np.ceil(x / self.__multiple_of) * self.__multiple_of).astype(int)

        return y

    def get_size(self, width, height):
        # determine new height and width
        scale_height = self.__height / height
        scale_width = self.__width / width

        if self.__keep_aspect_ratio:
            if self.__resize_method == "lower_bound":
                # scale such that output size is lower bound
                if scale_width > scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "upper_bound":
                # scale such that output size is upper bound
                if scale_width < scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "minimal":
                # scale as least as possbile
                if abs(1 - scale_width) < abs(1 - scale_height):
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            else:
                raise ValueError(f"resize_method {self.__resize_method} not implemented")

        if self.__resize_method == "lower_bound":
            new_height = self.constrain_to_multiple_of(scale_height * height, min_val=self.__height)
            new_width = self.constrain_to_multiple_of(scale_width * width, min_val=self.__width)
        elif self.__resize_method == "upper_bound":
            new_height = self.constrain_to_multiple_of(scale_height * height, max_val=self.__height)
            new_width = self.constrain_to_multiple_of(scale_width * width, max_val=self.__width)
        elif self.__resize_method == "minimal":
            new_height = self.constrain_to_multiple_of(scale_height * height)
            new_width = self.constrain_to_multiple_of(scale_width * width)
        else:
            raise ValueError(f"resize_method {self.__resize_method} not implemented")

        return (new_width, new_height)

    def __call__(self, sample):
        width, height = self.get_size(sample["image"].shape[1], sample["image"].shape[0])
        
        # resize sample
        sample["image"] = cv2.resize(sample["image"], (width, height), interpolation=self.__image_interpolation_method)

        if self.__resize_target:
            if "depth" in sample:
                sample["depth"] = cv2.resize(sample["depth"], (width, height), interpolation=cv2.INTER_NEAREST)
                
            if "mask" in sample:
                sample["mask"] = cv2.resize(sample["mask"].astype(np.float32), (width, height), interpolation=cv2.INTER_NEAREST)
        
        return sample


class NormalizeImage(object):
    """Normlize image by given mean and std.
    """

    def __init__(self, mean, std):
        self.__mean = mean
        self.__std = std

    def __call__(self, sample):
        sample["image"] = (sample["image"] - self.__mean) / self.__std

        return sample


class PrepareForNet(object):
    """Prepare sample for usage as network input.
    """

    def __init__(self):
        pass

    def __call__(self, sample):
        image = np.transpose(sample["image"], (2, 0, 1))
        sample["image"] = np.ascontiguousarray(image).astype(np.float32)

        if "depth" in sample:
            depth = sample["depth"].astype(np.float32)
            sample["depth"] = np.ascontiguousarray(depth)
        
        if "mask" in sample:
            sample["mask"] = sample["mask"].astype(np.float32)
            sample["mask"] = np.ascontiguousarray(sample["mask"])
        
        return sample


def _make_fusion_block(features, use_bn, size=None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


class ConvBlock(nn.Module):
    def __init__(self, in_feature, out_feature):
        super().__init__()
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_feature, out_feature, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_feature),
            nn.ReLU(True)
        )
    
    def forward(self, x):
        return self.conv_block(x)


class DPTHead(nn.Module):
    def __init__(
        self, 
        in_channels, 
        features=256,
        use_bn=False, 
        out_channels=[256, 512, 1024, 1024], 
        use_clstoken=False
    ):
        super(DPTHead, self).__init__()
        
        self.use_clstoken = use_clstoken
        
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])
        
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
        
        self.scratch.stem_transpose = None
        
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)
        
        head_features_1 = features
        head_features_2 = 32
        
        self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            nn.Identity(),
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


class DepthAnythingV2(nn.Module):
    def __init__(
        self, 
        encoder='vitl', 
        features=256, 
        out_channels=[256, 512, 1024, 1024], 
        use_bn=False, 
        use_clstoken=False
    ):
        super(DepthAnythingV2, self).__init__()
        
        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11], 
            'vitl': [4, 11, 17, 23], 
            'vitg': [9, 19, 29, 39]
        }

        self.encoder = encoder
        self.pretrained = DINOv2(model_name=encoder)
        
        self.depth_head = DPTHead(self.pretrained.embed_dim, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken)

    def forward(self, x):
        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14
        
        features = self.pretrained.get_intermediate_layers(x, self.intermediate_layer_idx[self.encoder], return_class_token=True)

        depth = self.depth_head(features, patch_h, patch_w)
        depth = F.relu(depth)

        return depth.squeeze(1)

    @torch.no_grad()
    def infer_image(self, raw_image, input_size=518):
        image, (h, w) = self.image2tensor(raw_image, input_size)
        
        depth = self.forward(image)
        
        depth = F.interpolate(depth[:, None], (h, w), mode="bilinear", align_corners=True)[0, 0]
        
        return depth.cpu().numpy()
    
    def image2tensor(self, raw_image, input_size=518):        
        transform = Compose([
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
        ])
        
        h, w = raw_image.shape[:2]
        
        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
        
        image = transform({'image': image})['image']
        image = torch.from_numpy(image).unsqueeze(0)
        
        DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        image = image.to(DEVICE)
        
        return image, (h, w)
    

class DepthAnythingV2LM(BaseRegressionTask):
    def __init__(self,
                 encoder:str,
                 learning_rate:float,
                 weight_decay:float,
                 lr_scheduler_patience:int,
                 lr_scheduler_factor:float,
                 metrics:Optional[List[torchmetrics.Metric]],
                 optimizer:Optional[Callable[[List[torch.nn.Parameter]], torch.optim.Optimizer]] = None,
                 scheduler:Optional[Callable[[torch.optim.Optimizer], Dict[str, Any]]] = None,
                 loss_fn:Optional[nn.Module] = None,
                 features:int = 256,
                 out_channels=[256, 512, 1024, 1024], 
                 use_bn:bool = False,
                 use_clstoken:bool = False,
                 ):
        # 先创建模型
        model = DepthAnythingV2(encoder=encoder, 
                               features=features, 
                               out_channels=out_channels, 
                               use_bn=use_bn, 
                               use_clstoken=use_clstoken)
        # 明确传递所有参数给父类
        super().__init__(
            model=model,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            lr_scheduler_patience=lr_scheduler_patience,
            lr_scheduler_factor=lr_scheduler_factor,
            metrics=metrics,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn
        )
        # 保存模型特定的超参数
        self.save_hyperparameters(ignore=['model', 'optimizer', 'scheduler', 'loss_fn', 'metrics'])
    
    def forward(self, input):
        return self.model(input)


if __name__ == "__main__":
    """
    用于测试DepthAnythingV2计算过程是否存在错误的主函数
    """
    import cv2
    import matplotlib.pyplot as plt
    import os
    import time
    import argparse
    
    parser = argparse.ArgumentParser(description='Test DepthAnythingV2 Model')
    parser.add_argument('--encoder', type=str, default='vitb', help='Encoder type: vits, vitb, vitl, vitg')
    parser.add_argument('--input_image', type=str, default=None, help='Path to input image (optional)')
    parser.add_argument('--input_size', type=int, default=518, help='Input image size (default: 518)')
    parser.add_argument('--out_dir', type=str, default='./', help='Output directory for debug visualizations')
    args = parser.parse_args()
    
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 创建模型
    embed_dims = {
        'vits': 384,
        'vitb': 768,
        'vitl': 1024,
        'vitg': 1536
    }
    
    encoder_type = args.encoder
    embed_dim = embed_dims[encoder_type]
    out_channels = [embed_dim, embed_dim, embed_dim, embed_dim]
    
    print(f"Creating DepthAnythingV2 with:")
    print(f"  - Encoder: {encoder_type}")
    print(f"  - Embed dim: {embed_dim}")
    print(f"  - Out channels: {out_channels}")
    
    model = DepthAnythingV2(
        encoder=encoder_type,
        features=256,
        out_channels=out_channels,
        use_bn=False,
        use_clstoken=False
    )
    model = model.to(device)
    model.eval()
    
    # 生成输入数据
    batch_size = 1
    image_size = args.input_size  # 确保是14的倍数
    
    # 如果指定了输入图像，则加载它
    if args.input_image and os.path.exists(args.input_image):
        print(f"Loading image from {args.input_image}")
        image, (orig_h, orig_w) = model.image2tensor(cv2.imread(args.input_image), input_size=image_size)
        input_tensor = image.to(device)
        is_real_image = True
    else:
        # 否则使用随机数据
        print("Using random input data")
        input_tensor = torch.randn(batch_size, 3, image_size, image_size).to(device)
        is_real_image = False
    
    try:
        # 计时
        start_time = time.time()
        
        # 1. 分解前向传播步骤进行测试
        print("\n==== 测试特征提取 ====")
        patch_h, patch_w = input_tensor.shape[-2] // 14, input_tensor.shape[-1] // 14
        print(f"Patch dimensions: h={patch_h}, w={patch_w}")
        
        with torch.no_grad():
            # 获取DINOv2特征
            features = model.pretrained.get_intermediate_layers(
                input_tensor, 
                model.intermediate_layer_idx[encoder_type], 
                return_class_token=True
            )
            
            print(f"提取了 {len(features)} 层特征")
            for i, feature in enumerate(features):
                feat, cls_token = feature
                print(f"  层 {i}:")
                print(f"    特征形状: {feat.shape}")
                print(f"    类别token形状: {cls_token.shape}")
                print(f"    特征统计: min={feat.min().item():.4f}, max={feat.max().item():.4f}, mean={feat.mean().item():.4f}")
            
            # 测试DPT头部
            print("\n==== 测试DPT头部 ====")
            # 测试投影层
            print("测试投影层:")
            projected_features = []
            for i, x in enumerate(features):
                x = x[0]  # 不使用类别token
                x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
                print(f"  特征 {i} 重塑后形状: {x.shape}")
                
                # 投影
                proj = model.depth_head.projects[i](x)
                print(f"  投影后形状: {proj.shape}")
                
                # 调整尺寸
                resized = model.depth_head.resize_layers[i](proj)
                print(f"  调整尺寸后形状: {resized.shape}")
                
                projected_features.append(resized)
            
            # 测试scratch网络
            print("\n测试Scratch网络:")
            layer_1, layer_2, layer_3, layer_4 = projected_features
            
            layer_1_rn = model.depth_head.scratch.layer1_rn(layer_1)
            layer_2_rn = model.depth_head.scratch.layer2_rn(layer_2)
            layer_3_rn = model.depth_head.scratch.layer3_rn(layer_3)
            layer_4_rn = model.depth_head.scratch.layer4_rn(layer_4)
            
            print(f"  layer_1_rn形状: {layer_1_rn.shape}")
            print(f"  layer_2_rn形状: {layer_2_rn.shape}")
            print(f"  layer_3_rn形状: {layer_3_rn.shape}")
            print(f"  layer_4_rn形状: {layer_4_rn.shape}")
            
            # 测试refinenet路径
            print("\n测试refinenet路径:")
            path_4 = model.depth_head.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
            print(f"  path_4形状: {path_4.shape}")
            
            path_3 = model.depth_head.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
            print(f"  path_3形状: {path_3.shape}")
            
            path_2 = model.depth_head.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
            print(f"  path_2形状: {path_2.shape}")
            
            path_1 = model.depth_head.scratch.refinenet1(path_2, layer_1_rn)
            print(f"  path_1形状: {path_1.shape}")
            
            # 测试输出卷积
            print("\n测试输出卷积:")
            out = model.depth_head.scratch.output_conv1(path_1)
            print(f"  第一输出卷积后形状: {out.shape}")
            
            out = F.interpolate(out, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True)
            print(f"  上采样后形状: {out.shape}")
            
            out = model.depth_head.scratch.output_conv2(out)
            print(f"  最终输出形状: {out.shape}")
            
            # 最终处理
            depth = F.relu(out)
            depth = depth.squeeze(1)
            print(f"  最终深度图形状: {depth.shape}")
            
            # 进行完整的前向传播
            print("\n==== 执行完整前向传播 ====")
            depth_final = model(input_tensor)
            print(f"最终深度估计形状: {depth_final.shape}")
            
            end_time = time.time()
            print(f"\n推理用时: {end_time - start_time:.4f} 秒")
            print(f"深度范围: min={depth_final.min().item():.4f}, max={depth_final.max().item():.4f}, mean={depth_final.mean().item():.4f}")
            
            # 如果是真实图像，保存可视化结果
            if is_real_image:
                # 可视化并保存结果
                output_dir = args.out_dir
                os.makedirs(output_dir, exist_ok=True)
                
                plt.figure(figsize=(12, 5))
                
                # 显示输入图像
                plt.subplot(1, 2, 1)
                input_image = input_tensor[0].permute(1, 2, 0).cpu().numpy()
                # 反归一化
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                input_image = std * input_image + mean
                input_image = np.clip(input_image, 0, 1)
                plt.imshow(input_image)
                plt.title('Input Image')
                plt.axis('off')
                
                # 显示深度图
                plt.subplot(1, 2, 2)
                depth_vis = depth_final[0].cpu().numpy()
                plt.imshow(depth_vis, cmap='plasma')
                plt.colorbar(label='Depth')
                plt.title('Estimated Depth')
                plt.axis('off')
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'depth_estimation_result.png'))
                print(f"结果已保存至 {os.path.join(output_dir, 'depth_estimation_result.png')}")
                
                # 保存深度图为彩色图
                depth_colored = plt.cm.plasma(depth_vis / np.max(depth_vis))
                depth_colored = (depth_colored[:, :, :3] * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(output_dir, 'depth_colored.png'), cv2.cvtColor(depth_colored, cv2.COLOR_RGB2BGR))
                
                # 保存为伪彩色可视化
                norm_depth = cv2.normalize(depth_vis, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                depth_color = cv2.applyColorMap(norm_depth, cv2.COLORMAP_INFERNO)
                cv2.imwrite(os.path.join(output_dir, 'depth_colormap.png'), depth_color)
            
            print("\n测试完成！所有计算步骤正常。")
                
    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()