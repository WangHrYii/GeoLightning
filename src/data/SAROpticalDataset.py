#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAR和Optical影像联合语义分割数据集
支持PyTorch Lightning集成
"""

import os
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split
from lightning import LightningDataModule
import cv2
from osgeo import gdal
from typing import Any, Dict, List, Optional, Tuple, Union
from PIL import Image
import warnings

# 配置GDAL
gdal.UseExceptions()  # 启用GDAL异常处理，消除警告
gdal.PushErrorHandler('CPLQuietErrorHandler')  # 抑制GDAL错误信息

# 忽略所有警告
warnings.filterwarnings("ignore")

# 如果需要只忽略特定警告，可以使用以下代码：
# warnings.filterwarnings("ignore", category=UserWarning)
# warnings.filterwarnings("ignore", category=FutureWarning)
# warnings.filterwarnings("ignore", message=".*gdal.*")


def read_image_with_gdal(filename: str) -> np.ndarray:
    """
    使用GDAL读取图像，返回ndarray
    
    Args:
        filename: 输入影像路径
        
    Returns:
        ndarray格式的影像数据，形状为 (bands, height, width)
    """
    dataset = gdal.Open(filename)
    if not dataset:
        raise FileNotFoundError(f"无法打开文件: {filename}")

    # 使用 ReadAsArray 读取所有波段的数据
    img = dataset.ReadAsArray()

    # 如果图像是单波段的，ReadAsArray 返回 (height, width)，需要增加一个维度
    if img.ndim == 2:
        img = np.expand_dims(img, axis=0)

    # 确保数据类型一致
    img = img.astype(np.float32)

    # 清理资源
    dataset = None
    
    return img


def read_image_with_pil(filename: str) -> np.ndarray:
    """
    使用PIL读取图像，返回ndarray
    
    Args:
        filename: 输入影像路径
        
    Returns:
        ndarray格式的影像数据，形状为 (height, width, channels)
    """
    img = Image.open(filename)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    img_array = np.array(img, dtype=np.float32)
    return img_array


def replace_invalid_values(data: np.ndarray, nodata_values: List[float] = [-9999, np.nan]) -> np.ndarray:
    """
    将数据中的Nodata值和NaN值替换为0
    
    Args:
        data: 输入数据
        nodata_values: 无效值列表
        
    Returns:
        处理后的数据
    """
    for nodata in nodata_values:
        if np.isnan(nodata):
            data[np.isnan(data)] = 0
        else:
            data[data == nodata] = 0
    return data


def normalize_sar_data(sar_data: np.ndarray, method: str = 'db') -> np.ndarray:
    """
    SAR数据归一化
    
    Args:
        sar_data: SAR数据
        method: 归一化方法 ('db', 'linear', 'percentile')
        
    Returns:
        归一化后的数据
    """
    if method == 'db':
        # 转换为dB
        sar_data = np.where(sar_data > 0, 10 * np.log10(sar_data + 1e-8), -80)
        # 裁剪到合理范围
        sar_data = np.clip(sar_data, -30, 10)
        # 归一化到0-1
        sar_data = (sar_data + 30) / 40
    elif method == 'linear':
        # 线性归一化
        sar_data = np.clip(sar_data, 0, np.percentile(sar_data, 99))
        sar_data = sar_data / (np.percentile(sar_data, 99) + 1e-8)
    elif method == 'percentile':
        # 百分位数归一化
        p2, p98 = np.percentile(sar_data, [2, 98])
        sar_data = np.clip(sar_data, p2, p98)
        sar_data = (sar_data - p2) / (p98 - p2 + 1e-8)
    
    return sar_data


class SAROpticalDataset(Dataset):
    """
    SAR和光学影像联合语义分割数据集
    
    数据组织结构:
    data_root/
    ├── optical/
    │   ├── image1.tif
    │   └── image2.tif
    ├── sar/
    │   ├── image1.tif
    │   └── image2.tif
    └── masks/
        ├── image1.tif
        └── image2.tif
    """
    
    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        optical_dir: str = '2_Opt',
        sar_dir: str = '1_SAR', 
        mask_dir: str = '0_Label',
        image_size: Tuple[int, int] = (512, 512),
        num_classes: int = 6,
        sar_channels: int = 1,
        optical_channels: int = 3,
        mask_values: Optional[List[int]] = [0, 1, 2, 3, 4, 5],
        sar_normalization: str = 'linear',
        optical_normalization: bool = False,
        scale: float = 1.0,
        transform: Optional[A.Compose] = None,
        use_gdal: bool = True,
        file_extensions: List[str] = ['.tif', '.tiff', '.png', '.jpg'],
        enable_cache: bool = False  # 是否启用内存缓存（仅推荐小数据集使用）
    ):
        """
        初始化数据集
        
        Args:
            data_root: 数据根目录
            split: 数据集划分 ('train', 'val', 'test')
            optical_dir: 光学影像目录名
            sar_dir: SAR影像目录名
            mask_dir: 掩码目录名
            image_size: 图像尺寸 (height, width)
            num_classes: 类别数量
            sar_channels: SAR数据通道数
            optical_channels: 光学数据通道数
            mask_values: 掩码值列表，如果为None则自动生成[0, 1, ..., num_classes-1]
            sar_normalization: SAR数据归一化方法
            optical_normalization: 是否对光学数据进行ImageNet归一化
            scale: 图像缩放因子
            transform: 数据增强变换
            use_gdal: 是否使用GDAL读取数据
            file_extensions: 支持的文件扩展名
        """
        self.data_root = Path(data_root)
        self.split = split
        self.optical_dir = self.data_root / optical_dir
        self.sar_dir = self.data_root / sar_dir
        self.mask_dir = self.data_root / mask_dir
        
        self.image_size = image_size
        self.num_classes = num_classes
        self.sar_channels = sar_channels
        self.optical_channels = optical_channels
        self.mask_values = mask_values if mask_values is not None else list(range(num_classes))
        self.sar_normalization = sar_normalization
        self.optical_normalization = optical_normalization
        self.scale = scale
        self.transform = transform
        self.use_gdal = use_gdal
        self.file_extensions = file_extensions
        self.enable_cache = enable_cache
        
        # 内存缓存（仅在enable_cache=True时使用）
        self._cache = {} if enable_cache else None
        
        # 验证目录存在
        for dir_path in [self.optical_dir, self.sar_dir, self.mask_dir]:
            if not dir_path.exists():
                raise ValueError(f"目录不存在: {dir_path}")
        
        # 收集文件列表
        self._collect_files()
        
        print(f"初始化{split}数据集，共{len(self.files)}个样本")
    
    def _collect_files(self):
        """收集所有有效的文件组合"""
        self.files = []
        
        # 获取光学影像文件列表
        optical_files = {}
        for ext in self.file_extensions:
            for file_path in self.optical_dir.glob(f"*{ext}"):
                basename = file_path.stem
                optical_files[basename] = file_path
        
        # 检查对应的SAR和mask文件是否存在
        for basename, optical_path in optical_files.items():
            sar_path = None
            mask_path = None
            
            # 查找对应的SAR文件
            for ext in self.file_extensions:
                potential_sar = self.sar_dir / f"{basename}{ext}"
                if potential_sar.exists():
                    sar_path = potential_sar
                    break
            
            # 查找对应的mask文件
            for ext in self.file_extensions:
                potential_mask = self.mask_dir / f"{basename}{ext}"
                if potential_mask.exists():
                    mask_path = potential_mask
                    break
            
            # 只有当三个文件都存在时才添加到列表中
            if sar_path and mask_path:
                self.files.append({
                    'optical': optical_path,
                    'sar': sar_path,
                    'mask': mask_path,
                    'basename': basename
                })
        
        if not self.files:
            raise RuntimeError(f"在{self.data_root}中未找到有效的文件组合")
    
    def __len__(self) -> int:
        return len(self.files)
    
    def _load_image(self, path: Path, is_mask: bool = False) -> np.ndarray:
        """加载图像"""
        if self.use_gdal or path.suffix.lower() in ['.tif', '.tiff']:
            img = read_image_with_gdal(str(path))
            # GDAL返回(C, H, W)格式，转换为(H, W, C)
            if img.ndim == 3:
                img = img.transpose(1, 2, 0)
            elif img.ndim == 2:
                img = np.expand_dims(img, axis=-1)
        else:
            img = read_image_with_pil(str(path))
            if img.ndim == 2:
                img = np.expand_dims(img, axis=-1)
        
        return img
    
    def _preprocess_image(self, img: np.ndarray, is_optical: bool = True, is_mask: bool = False) -> np.ndarray:
        """预处理图像"""
        if is_mask:
            # 处理掩码
            if img.ndim == 3:
                img = img[:, :, 0]  # 取第一个通道
            
            # 创建新的掩码
            mask = np.zeros_like(img, dtype=np.int64)
            for i, value in enumerate(self.mask_values):
                mask[img == value] = i
            
            return mask
        
        else:
            # 处理图像数据
            img = replace_invalid_values(img, [-9999, np.nan])
            
            # 确保图像是3D的 (H, W, C)
            if img.ndim == 2:
                img = np.expand_dims(img, axis=-1)
            
            if is_optical:
                # 光学数据处理
                if img.shape[-1] > self.optical_channels:
                    img = img[:, :, :self.optical_channels]
                elif img.shape[-1] < self.optical_channels:
                    # 如果通道数不足，重复最后一个通道
                    pad_channels = self.optical_channels - img.shape[-1]
                    last_channel = img[:, :, -1:].repeat(pad_channels, axis=-1)
                    img = np.concatenate([img, last_channel], axis=-1)
                
                # 归一化到0-255
                if img.max() <= 1.0:
                    img = img * 255.0
                img = np.clip(img, 0, 255)
                
            else:
                # SAR数据处理
                if img.shape[-1] > self.sar_channels:
                    img = img[:, :, :self.sar_channels]
                elif img.shape[-1] < self.sar_channels:
                    # 如果通道数不足，重复最后一个通道
                    pad_channels = self.sar_channels - img.shape[-1]
                    last_channel = img[:, :, -1:].repeat(pad_channels, axis=-1)
                    img = np.concatenate([img, last_channel], axis=-1)

                if img.ndim == 2:
                    img = np.expand_dims(img, axis=-1)
            
            return img.astype(np.float32)
    
    def _resize_image(self, img: np.ndarray, is_mask: bool = False) -> np.ndarray:
        """调整图像尺寸"""
        target_h, target_w = self.image_size
        
        if self.scale != 1.0:
            target_h = int(target_h * self.scale)
            target_w = int(target_w * self.scale)
        
        # # 确保尺寸是14的倍数（适用于某些模型）
        # target_h = int(np.ceil(target_h / 14) * 14)
        # target_w = int(np.ceil(target_w / 14) * 14)
        
        if is_mask:
            interpolation = cv2.INTER_NEAREST
        else:
            interpolation = cv2.INTER_LINEAR
        
        if img.ndim == 2:
            resized = cv2.resize(img, (target_w, target_h), interpolation=interpolation)
        else:
            resized = cv2.resize(img, (target_w, target_h), interpolation=interpolation)
        
        # 确保resize后的图像保持正确的维度
        if not is_mask and resized.ndim == 2:
            resized = np.expand_dims(resized, axis=-1)
        
        return resized
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取单个样本"""
        file_info = self.files[idx]
        
        # 检查缓存
        cache_key = file_info['basename']
        if self._cache is not None and cache_key in self._cache:
            optical_img, sar_img, mask = self._cache[cache_key]
        else:
            # 加载图像
            optical_img = self._load_image(file_info['optical'])
            sar_img = self._load_image(file_info['sar'])
            mask = self._load_image(file_info['mask'], is_mask=True)
            
            # 预处理
            optical_img = self._preprocess_image(optical_img, is_optical=True)
            sar_img = self._preprocess_image(sar_img, is_optical=False)
            mask = self._preprocess_image(mask, is_mask=True)
            
            # 调整尺寸
            optical_img = self._resize_image(optical_img)
            sar_img = self._resize_image(sar_img)
            mask = self._resize_image(mask, is_mask=True)
            
            # 缓存预处理后的数据
            if self._cache is not None:
                self._cache[cache_key] = (optical_img.copy(), sar_img.copy(), mask.copy())
        
        # 确保mask是2D
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        
        # 确保SAR和光学图像都是3D的 (H, W, C)
        if optical_img.ndim == 2:
            optical_img = np.expand_dims(optical_img, axis=-1)
        if sar_img.ndim == 2:
            sar_img = np.expand_dims(sar_img, axis=-1)
        
        # 数据增强
        if self.transform is not None:
            # 合并optical和sar为一个图像进行同步变换
            combined_img = np.concatenate([optical_img, sar_img], axis=-1)
            
            transformed = self.transform(image=combined_img, mask=mask)
            combined_img = transformed['image']
            mask = transformed['mask']
            
            # 分离optical和sar
            optical_img = combined_img[:self.optical_channels, :, :]
            sar_img = combined_img[self.optical_channels:, :, :]
        else:
            # 默认转换为tensor
            to_tensor = A.Compose([ToTensorV2()])
            
            optical_img = to_tensor(image=optical_img)['image']
            sar_img = to_tensor(image=sar_img)['image']
            mask = torch.from_numpy(mask).long()
        
        # 光学数据归一化
        if self.optical_normalization:
            mean = torch.tensor([0.485, 0.456, 0.406])[:optical_img.shape[0]].view(-1, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225])[:optical_img.shape[0]].view(-1, 1, 1)
            optical_img = (optical_img / 255.0 - mean) / std
        else:
            optical_img = optical_img / 255.0
        
        # SAR数据归一化（已经在预处理中完成，这里只需要缩放到0-1）
        sar_img = sar_img / 255.0
        
        return {
            'optical': optical_img,
            'sar': sar_img,
            'mask': mask,
            'basename': file_info['basename']
        }


class SAROpticalDataModule(LightningDataModule):
    """
    SAR和光学影像联合语义分割数据模块
    """
    
    def __init__(
        self,
        data_root: str,
        optical_dir: str = '2_Opt',
        sar_dir: str = '1_SAR',
        mask_dir: str = '0_Label',
        image_size: Tuple[int, int] = (512, 512),
        num_classes: int = 6,
        sar_channels: int = 1,
        optical_channels: int = 3,
        batch_size: int = 4,
        num_workers: int = 4,
        val_split: float = 0.2,
        test_split: float = 0.1,
        mask_values: Optional[List[int]] = [0, 1, 2, 3, 4, 5],
        sar_normalization: str = 'linear',
        optical_normalization: bool = False,
        scale: float = 1.0,
        augmentation_config: Optional[Dict] = None,
        use_gdal: bool = True,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: int = 4
    ):
        """
        初始化数据模块
        
        Args:
            data_root: 数据根目录
            optical_dir: 光学影像目录名
            sar_dir: SAR影像目录名  
            mask_dir: 掩码目录名
            image_size: 图像尺寸
            num_classes: 类别数量
            sar_channels: SAR通道数
            optical_channels: 光学通道数
            batch_size: 批次大小
            num_workers: 数据加载工作线程数
            val_split: 验证集比例
            test_split: 测试集比例
            mask_values: 掩码值列表
            sar_normalization: SAR归一化方法
            optical_normalization: 是否对光学数据进行归一化
            scale: 图像缩放因子
            augmentation_config: 数据增强配置
            use_gdal: 是否使用GDAL
            pin_memory: 是否使用pin_memory
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.data_root = data_root
        self.optical_dir = optical_dir
        self.sar_dir = sar_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.num_classes = num_classes
        self.sar_channels = sar_channels
        self.optical_channels = optical_channels
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.test_split = test_split
        self.mask_values = mask_values
        self.sar_normalization = sar_normalization
        self.optical_normalization = optical_normalization
        self.scale = scale
        self.use_gdal = use_gdal
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor
        
        # 设置数据增强
        self.train_transform = self._get_train_transform(augmentation_config)
        self.val_transform = self._get_val_transform()
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def _get_train_transform(self, config: Optional[Dict] = None) -> A.Compose:
        """获取训练时的数据增强"""
        if config is None:
            config = {
                'horizontal_flip': 0.5,
                'vertical_flip': 0.5,
                'rotation': 15,
                'brightness': 0.2,
                'contrast': 0.2,
                'gamma': (80, 120),
                'elastic': True
            }
        
        transforms = []
        
        if config.get('horizontal_flip', 0) > 0:
            transforms.append(A.HorizontalFlip(p=config['horizontal_flip']))
        
        if config.get('vertical_flip', 0) > 0:
            transforms.append(A.VerticalFlip(p=config['vertical_flip']))
        
        if config.get('rotation', 0) > 0:
            transforms.append(A.Rotate(limit=config['rotation'], p=0.5))
        
        if config.get('brightness', 0) > 0 or config.get('contrast', 0) > 0:
            transforms.append(A.RandomBrightnessContrast(
                brightness_limit=config.get('brightness', 0),
                contrast_limit=config.get('contrast', 0),
                p=0.5
            ))
        
        if config.get('gamma'):
            gamma_range = config['gamma']
            transforms.append(A.RandomGamma(gamma_limit=gamma_range, p=0.5))
        
        if config.get('elastic', False):
            transforms.append(A.ElasticTransform(p=0.3))
        
        transforms.append(ToTensorV2())
        
        return A.Compose(transforms)
    
    def _get_val_transform(self) -> A.Compose:
        """获取验证时的数据变换"""
        return A.Compose([ToTensorV2()])
    
    def setup(self, stage: Optional[str] = None):
        """设置数据集"""
        # 创建完整数据集
        full_dataset = SAROpticalDataset(
            data_root=self.data_root,
            optical_dir=self.optical_dir,
            sar_dir=self.sar_dir,
            mask_dir=self.mask_dir,
            image_size=self.image_size,
            num_classes=self.num_classes,
            sar_channels=self.sar_channels,
            optical_channels=self.optical_channels,
            mask_values=self.mask_values,
            sar_normalization=self.sar_normalization,
            optical_normalization=self.optical_normalization,
            scale=self.scale,
            use_gdal=self.use_gdal
        )
        
        # 计算分割大小
        total_size = len(full_dataset)
        test_size = int(total_size * self.test_split)
        val_size = int(total_size * self.val_split)
        train_size = total_size - val_size - test_size
        
        # 随机分割数据集
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset, 
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # 为训练、验证、测试集创建独立的数据集实例，避免重复I/O
        # 训练集数据集（带数据增强）
        self.train_dataset = SAROpticalDataset(
            data_root=self.data_root,
            split='train',
            optical_dir=self.optical_dir,
            sar_dir=self.sar_dir,
            mask_dir=self.mask_dir,
            image_size=self.image_size,
            num_classes=self.num_classes,
            sar_channels=self.sar_channels,
            optical_channels=self.optical_channels,
            mask_values=self.mask_values,
            sar_normalization=self.sar_normalization,
            optical_normalization=self.optical_normalization,
            scale=self.scale,
            transform=self.train_transform,
            use_gdal=self.use_gdal
        )
        
        # 验证集数据集（无数据增强）
        self.val_dataset = SAROpticalDataset(
            data_root=self.data_root,
            split='val',
            optical_dir=self.optical_dir,
            sar_dir=self.sar_dir,
            mask_dir=self.mask_dir,
            image_size=self.image_size,
            num_classes=self.num_classes,
            sar_channels=self.sar_channels,
            optical_channels=self.optical_channels,
            mask_values=self.mask_values,
            sar_normalization=self.sar_normalization,
            optical_normalization=self.optical_normalization,
            scale=self.scale,
            transform=self.val_transform,
            use_gdal=self.use_gdal
        )
        
        # 测试集数据集（无数据增强）
        self.test_dataset = SAROpticalDataset(
            data_root=self.data_root,
            split='test',
            optical_dir=self.optical_dir,
            sar_dir=self.sar_dir,
            mask_dir=self.mask_dir,
            image_size=self.image_size,
            num_classes=self.num_classes,
            sar_channels=self.sar_channels,
            optical_channels=self.optical_channels,
            mask_values=self.mask_values,
            sar_normalization=self.sar_normalization,
            optical_normalization=self.optical_normalization,
            scale=self.scale,
            transform=self.val_transform,
            use_gdal=self.use_gdal
        )
        
        # 使用预先计算的索引来分割数据集
        total_files = len(full_dataset.files)
        indices = list(range(total_files))
        
        # 使用相同的随机种子确保一致的分割
        np.random.seed(42)
        np.random.shuffle(indices)
        
        test_size = int(total_files * self.test_split)
        val_size = int(total_files * self.val_split)
        train_size = total_files - val_size - test_size
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        # 设置每个数据集的文件列表
        self.train_dataset.files = [full_dataset.files[i] for i in train_indices]
        self.val_dataset.files = [full_dataset.files[i] for i in val_indices]
        self.test_dataset.files = [full_dataset.files[i] for i in test_indices]
        
        print(f"数据集分割完成 - 训练: {len(self.train_dataset)}, "
              f"验证: {len(self.val_dataset)}, 测试: {len(self.test_dataset)}")
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            drop_last=True
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None
        )
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None
        )


if __name__ == "__main__":
    # 使用示例
    data_module = SAROpticalDataModule(
        data_root="data/RSIPAC_25_T1/train",
        image_size=(512, 512),
        num_classes=6,
        batch_size=4,
        num_workers=4,
        val_split=0.2,
        test_split=0.1
    )
    
    data_module.setup()
    
    # 测试数据加载
    train_loader = data_module.train_dataloader()
    for batch in train_loader:
        optical = batch['optical']
        sar = batch['sar']
        mask = batch['mask']
        
        print(f"Optical shape: {optical.shape}")
        print(f"SAR shape: {sar.shape}")
        print(f"Mask shape: {mask.shape}")
        print(f"Mask unique values: {torch.unique(mask)}")
        break
