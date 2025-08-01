#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAR-Optical语义分割推理数据集（处理已分割的切片）
支持从两个文件夹中读取对应的SAR和光学切片
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from lightning import LightningDataModule
from osgeo import gdal
from pathlib import Path
from typing import Optional, List, Tuple, Dict
import warnings
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 配置GDAL
gdal.UseExceptions()
gdal.PushErrorHandler('CPLQuietErrorHandler')
warnings.filterwarnings("ignore")


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


def normalize_sar_data(sar_data: np.ndarray, method: str = 'linear') -> np.ndarray:
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


class SAROpticalTiledInferenceDataset(Dataset):
    """
    SAR-Optical语义分割推理数据集（处理已分割的切片）
    从两个文件夹中读取对应的SAR和光学切片
    """
    
    def __init__(
        self,
        sar_dir: str,
        optical_dir: str,
        image_size: Tuple[int, int] = (512, 512),
        sar_channels: int = 1,
        optical_channels: int = 3,
        sar_normalization: str = 'linear',
        optical_normalization: bool = False,
        file_extensions: List[str] = ['.tif', '.tiff', '.png', '.jpg']
    ):
        """
        初始化推理数据集
        
        Args:
            sar_dir: SAR切片文件夹路径
            optical_dir: 光学切片文件夹路径
            image_size: 图像尺寸 (height, width)
            sar_channels: SAR通道数
            optical_channels: 光学通道数
            sar_normalization: SAR归一化方法
            optical_normalization: 是否对光学数据进行ImageNet归一化
            file_extensions: 支持的文件扩展名
        """
        super().__init__()
        
        self.sar_dir = Path(sar_dir)
        self.optical_dir = Path(optical_dir)
        self.image_size = image_size
        self.sar_channels = sar_channels
        self.optical_channels = optical_channels
        self.sar_normalization = sar_normalization
        self.optical_normalization = optical_normalization
        self.file_extensions = file_extensions
        
        # 验证目录存在
        if not self.sar_dir.exists():
            raise ValueError(f"SAR目录不存在: {self.sar_dir}")
        if not self.optical_dir.exists():
            raise ValueError(f"光学影像目录不存在: {self.optical_dir}")
        
        # 收集文件对
        self.file_pairs = self._collect_file_pairs()
        
        # 设置数据变换
        self.transform = A.Compose([ToTensorV2()])
        
        print(f"初始化推理数据集，共{len(self.file_pairs)}个切片对")
    
    def _collect_file_pairs(self) -> List[Dict[str, Path]]:
        """收集所有有效的文件对"""
        file_pairs = []
        
        # 获取SAR文件列表
        sar_files = {}
        for ext in self.file_extensions:
            for file_path in self.sar_dir.glob(f"*{ext}"):
                basename = file_path.stem
                sar_files[basename] = file_path
        
        # 检查对应的光学文件是否存在
        for basename, sar_path in sar_files.items():
            optical_path = None
            
            # 查找对应的光学文件
            for ext in self.file_extensions:
                potential_optical = self.optical_dir / f"{basename}{ext}"
                if potential_optical.exists():
                    optical_path = potential_optical
                    break
            
            # 只有当两个文件都存在时才添加到列表中
            if optical_path:
                file_pairs.append({
                    'sar': sar_path,
                    'optical': optical_path,
                    'basename': basename
                })
        
        if not file_pairs:
            raise RuntimeError(f"在{self.sar_dir}和{self.optical_dir}中未找到匹配的文件对")
        
        # 按文件名排序，确保输出顺序一致
        file_pairs.sort(key=lambda x: x['basename'])
        
        return file_pairs
    
    def __len__(self) -> int:
        return len(self.file_pairs)
    
    def _load_image(self, path: Path) -> np.ndarray:
        """加载图像"""
        img = read_image_with_gdal(str(path))
        # GDAL返回(C, H, W)格式，转换为(H, W, C)
        if img.ndim == 3:
            img = img.transpose(1, 2, 0)
        elif img.ndim == 2:
            img = np.expand_dims(img, axis=-1)
        
        return img
    
    def _preprocess_image(self, img: np.ndarray, is_optical: bool = True) -> np.ndarray:
        """预处理图像"""
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
            
            # SAR数据归一化
            for c in range(img.shape[-1]):
                img[:, :, c] = normalize_sar_data(img[:, :, c], method=self.sar_normalization)
            
            # 归一化到0-255范围，与光学数据保持一致
            img = img * 255.0
        
        return img.astype(np.float32)
    
    def _resize_image(self, img: np.ndarray) -> np.ndarray:
        """调整图像尺寸"""
        target_h, target_w = self.image_size
        
        if img.shape[:2] != (target_h, target_w):
            import cv2
            resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            
            # 确保resize后的图像保持正确的维度
            if resized.ndim == 2:
                resized = np.expand_dims(resized, axis=-1)
            
            return resized
        
        return img
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取单个样本"""
        file_info = self.file_pairs[idx]
        
        # 加载图像
        optical_img = self._load_image(file_info['optical'])
        sar_img = self._load_image(file_info['sar'])
        
        # 预处理
        optical_img = self._preprocess_image(optical_img, is_optical=True)
        sar_img = self._preprocess_image(sar_img, is_optical=False)
        
        # 调整尺寸
        optical_img = self._resize_image(optical_img)
        sar_img = self._resize_image(sar_img)
        
        # 确保SAR和光学图像都是3D的 (H, W, C)
        if optical_img.ndim == 2:
            optical_img = np.expand_dims(optical_img, axis=-1)
        if sar_img.ndim == 2:
            sar_img = np.expand_dims(sar_img, axis=-1)
        
        # 应用变换（转换为tensor）
        optical_img = self.transform(image=optical_img)['image']
        sar_img = self.transform(image=sar_img)['image']
        
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
            'basename': file_info['basename'],
            'sar_path': str(file_info['sar']),
            'optical_path': str(file_info['optical'])
        }


class SAROpticalTiledInferenceDataModule(LightningDataModule):
    """
    SAR-Optical推理数据模块（处理已分割的切片）
    """
    
    def __init__(
        self,
        sar_dir: str,
        optical_dir: str,
        image_size: Tuple[int, int] = (512, 512),
        batch_size: int = 8,
        num_workers: int = 4,
        sar_channels: int = 1,
        optical_channels: int = 3,
        sar_normalization: str = 'linear',
        optical_normalization: bool = False,
        file_extensions: List[str] = ['.tif', '.tiff', '.png', '.jpg']
    ):
        super().__init__()
        self.sar_dir = sar_dir
        self.optical_dir = optical_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sar_channels = sar_channels
        self.optical_channels = optical_channels
        self.sar_normalization = sar_normalization
        self.optical_normalization = optical_normalization
        self.file_extensions = file_extensions
        
    def setup(self, stage: Optional[str] = None):
        self.dataset = SAROpticalTiledInferenceDataset(
            sar_dir=self.sar_dir,
            optical_dir=self.optical_dir,
            image_size=self.image_size,
            sar_channels=self.sar_channels,
            optical_channels=self.optical_channels,
            sar_normalization=self.sar_normalization,
            optical_normalization=self.optical_normalization,
            file_extensions=self.file_extensions
        )
        
    def predict_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True
        ) 