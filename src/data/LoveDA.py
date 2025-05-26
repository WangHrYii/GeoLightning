import os
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
import torchvision.transforms as T
from lightning import LightningDataModule

class LoveDADataset(Dataset):
    """LoveDA数据集加载类"""
    
    def __init__(
        self, 
        root_dir: str, 
        split: str = 'train', 
        vfm_size: int = 224, 
        original_size: int = 1024, 
        normalize: bool = False,
        image_extensions: List[str] = ['.png', '.jpg', '.jpeg']
    ):
        """
        LoveDA数据集
        
        参数:
            root_dir: 数据集根目录
            split: 数据集划分，'train' 或 'val'
            vfm_size: VFM模型输入尺寸(默认224)
            original_size: 原始图像尺寸(默认1024)
            normalize: 是否对图像进行标准化
            image_extensions: 支持的图像文件扩展名列表
        """
        self.root_dir = root_dir
        self.split = split.lower()  # 转换为小写，支持大小写不敏感
        self.vfm_size = vfm_size
        self.original_size = original_size
        self.normalize = normalize
        self.image_extensions = image_extensions

        
        # 检查split参数
        if self.split not in ['train', 'val', 'test']:
            raise ValueError(f"split参数必须为'train'、'val'或'test'，当前为{split}")
        
        # 设置区域类型
        self.area_types = ['Urban', 'Rural']
        
        # 收集所有图像文件路径
        self.img_files = []
        
        # 根据split处理不同区域的图像
        for area in self.area_types:
            if self.split == 'train':
                img_dir = os.path.join(root_dir, 'Train', area, 'images_png')
            elif self.split == 'val':
                img_dir = os.path.join(root_dir, 'Val', area, 'images_png')
            else:  # test
                img_dir = os.path.join(root_dir, 'Test', area, 'images_png')
            
            if os.path.exists(img_dir):
                # 添加当前区域的所有图像文件
                for file_name in os.listdir(img_dir):
                    if any(file_name.endswith(ext) for ext in self.image_extensions):
                        self.img_files.append((os.path.join(img_dir, file_name), area))
        
        # 排序确保结果可重现
        self.img_files.sort(key=lambda x: x[0])
        
        # 创建用于构建转换流程的函数
        def create_transform(target_size: int) -> T.Compose:
            transforms = [
                T.Resize((target_size, target_size), interpolation=T.InterpolationMode.BICUBIC),
                T.ToTensor()
            ]
            if normalize:
                transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
            return T.Compose(transforms)
        
        # 应用转换函数创建两种尺寸的转换
        self.vfm_transform = create_transform(self.vfm_size)
        self.original_transform = create_transform(self.original_size)
    
    def __len__(self) -> int:
        return len(self.img_files)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # 获取图像路径
        img_path, area = self.img_files[idx]
        
        # 读取图像
        image = Image.open(img_path).convert('RGB')
        
        # 应用两种转换
        vfm_image = self.vfm_transform(image)  # 224x224尺寸
        original_image = self.original_transform(image)  # 1024x1024尺寸
        
        return {
            'image': vfm_image,              # VFM输入图像(224x224)
            'original_image': original_image,  # 原始高分辨率图像(1024x1024)
            'path': img_path,
            'area': area
        }

class LoveDADatasetLD(LightningDataModule):
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        batch_size: int = 2,
        num_workers: int = 4,
        vfm_size: int = 224,
        original_size: int = 1024,
        normalize: bool = False,
        max_train_samples: Optional[int] = None,
        max_val_samples: Optional[int] = None,
        image_extensions: List[str] = ['.png', '.jpg', '.jpeg']
    ):
        """
        LoveDA数据集加载器
        
        参数:
            root_dir: 数据集根目录
            split: 数据集划分，'train' 或 'val'
            batch_size: 批次大小
            num_workers: 数据加载工作线程数
            vfm_size: VFM模型输入尺寸(默认224)
            original_size: 原始图像尺寸(默认1024)
            normalize: 是否对图像进行标准化
            max_train_samples: 最大训练样本数，None表示使用所有样本
            max_val_samples: 最大验证样本数，None表示使用所有样本
            image_extensions: 支持的图像文件扩展名列表
        """
        super().__init__()
        self.root_dir = root_dir
        self.split = split.lower()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.vfm_size = vfm_size
        self.original_size = original_size
        self.normalize = normalize
        self.max_train_samples = max_train_samples
        self.max_val_samples = max_val_samples
        self.image_extensions = image_extensions

        self.train_dataset = None
        self.val_dataset = None
        self.train_loader = None
        self.val_loader = None

    def setup(self, stage: Optional[str] = None):
        """准备数据集，这个方法会被Lightning在每个GPU上调用"""
        train_dataset = LoveDADataset(
            root_dir=self.root_dir, 
            split='train', 
            vfm_size=self.vfm_size, 
            original_size=self.original_size,
            normalize=self.normalize,
            image_extensions=self.image_extensions
        )
        
        val_dataset = LoveDADataset(
            root_dir=self.root_dir, 
            split='test', 
            vfm_size=self.vfm_size, 
            original_size=self.original_size,
            normalize=self.normalize,
            image_extensions=self.image_extensions
        )
        
        # 如果设置了最大样本数，则限制数据集大小
        if self.max_train_samples is not None:
            train_indices = list(range(min(self.max_train_samples, len(train_dataset))))
            train_dataset = Subset(train_dataset, train_indices)
        
        if self.max_val_samples is not None:
            val_indices = list(range(min(self.max_val_samples, len(val_dataset))))
            val_dataset = Subset(val_dataset, val_indices)
        
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,   
            pin_memory=True
        )
