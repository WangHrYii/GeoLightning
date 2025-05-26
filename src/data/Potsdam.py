import os
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
import torchvision.transforms as T
from lightning import LightningDataModule

    
class PotsdamDataset(Dataset):
    """Potsdam数据集加载类"""
    
    def __init__(
        self, 
        root_dir: str, 
        split: str = 'train', 
        vfm_size: int = 224, 
        original_size: int = 1024, 
        normalize: bool = True,
        image_extensions: List[str] = ['.png']
    ):
        """
        Potsdam数据集
        
        参数:
            root_dir: 数据集根目录
            split: 数据集划分，'train' 或 'test'
            vfm_size: VFM模型输入尺寸(默认224)
            original_size: 原始图像尺寸(默认1024)
            normalize: 是否对图像进行标准化
            image_extensions: 支持的图像文件扩展名列表
        """
        self.root_dir = root_dir
        self.split = split.lower()
        self.vfm_size = vfm_size
        self.original_size = original_size
        self.normalize = normalize
        self.image_extensions = image_extensions
        
        # 检查split参数
        if self.split not in ['train', 'test']:
            raise ValueError(f"split参数必须为'train'或'test'，当前为{split}")
        
        # 收集所有图像文件路径
        self.img_files = []
        self.mask_files = []
        
        # 设置图像和掩码目录
        img_dir = os.path.join(root_dir, split, 'images_png')
        mask_dir = os.path.join(root_dir, split, 'masks_png')
        
        if os.path.exists(img_dir) and os.path.exists(mask_dir):
            # 添加所有图像文件
            for file_name in os.listdir(img_dir):
                if any(file_name.endswith(ext) for ext in self.image_extensions):
                    img_path = os.path.join(img_dir, file_name)
                    mask_path = os.path.join(mask_dir, file_name)
                    if os.path.exists(mask_path):
                        self.img_files.append(img_path)
                        self.mask_files.append(mask_path)
        
        # 排序确保结果可重现
        self.img_files.sort()
        self.mask_files.sort()
        
        # 创建用于构建转换流程的函数
        def create_transform(target_size: int, is_vfm: bool = False) -> T.Compose:
            transforms = [
                T.Resize((target_size, target_size), 
                        interpolation=T.InterpolationMode.NEAREST if is_vfm else T.InterpolationMode.BICUBIC),
                T.ToTensor()
            ]
            if normalize and  is_vfm:
                transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
            return T.Compose(transforms)
        
        # 应用转换函数创建两种尺寸的转换
        self.vfm_transform = create_transform(self.vfm_size, is_vfm=True)
        self.original_transform = create_transform(self.original_size)
        self.mask_transform = create_transform(self.vfm_size)
    
    def __len__(self) -> int:
        return len(self.img_files)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # 获取图像和掩码路径
        img_path = self.img_files[idx]
        mask_path = self.mask_files[idx]
        
        # 读取图像和掩码
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # 转换为灰度图
        
        # 应用转换
        vfm_image = self.vfm_transform(image)  # 224x224尺寸
        original_image = self.original_transform(image)  # 1024x1024尺寸
        mask = self.mask_transform(mask)  # 1024x1024尺寸
        
        return {
            'image': vfm_image,              # VFM输入图像(224x224)
            'original_image': original_image,  # 原始高分辨率图像(1024x1024)
            'mask': mask,                    # 对应的掩码(1024x1024)
            'path': img_path
        }

class PotsdamDatasetLD(LightningDataModule):
    def __init__(
        self,
        root_dir: str,
        batch_size: int = 2,
        num_workers: int = 4,
        vfm_size: int = 224,
        original_size: int = 1024,
        normalize: bool = True,
        max_train_samples: Optional[int] = None,
        max_test_samples: Optional[int] = None,
        image_extensions: List[str] = ['.png']
    ):
        """
        Potsdam数据集加载器
        
        参数:
            root_dir: 数据集根目录
            batch_size: 批次大小
            num_workers: 数据加载工作线程数
            vfm_size: VFM模型输入尺寸(默认224)
            original_size: 原始图像尺寸(默认1024)
            normalize: 是否对图像进行标准化
            max_train_samples: 最大训练样本数，None表示使用所有样本
            max_test_samples: 最大测试样本数，None表示使用所有样本
            image_extensions: 支持的图像文件扩展名列表
        """
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.vfm_size = vfm_size
        self.original_size = original_size
        self.normalize = normalize
        self.max_train_samples = max_train_samples
        self.max_test_samples = max_test_samples
        self.image_extensions = image_extensions

        self.train_dataset = None
        self.test_dataset = None
        self.train_loader = None
        self.test_loader = None

    def setup(self, stage: Optional[str] = None):
        """准备数据集，这个方法会被Lightning在每个GPU上调用"""
        train_dataset = PotsdamDataset(
            root_dir=self.root_dir, 
            split='train', 
            vfm_size=self.vfm_size, 
            original_size=self.original_size,
            normalize=self.normalize,
            image_extensions=self.image_extensions
        )
        
        test_dataset = PotsdamDataset(
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
        
        if self.max_test_samples is not None:
            test_indices = list(range(min(self.max_test_samples, len(test_dataset))))
            test_dataset = Subset(test_dataset, test_indices)
        
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,   
            pin_memory=True
        )
        
    def val_dataloader(self):
        """使用test_dataloader作为验证集"""
        return self.test_dataloader()

if __name__ == "__main__":
    # 测试数据集加载
    root_dir = "/mnt/f/data/ISPRS semantic label data/Potsdam/processed"
    
    # 创建数据集实例
    dataset = PotsdamDataset(
        root_dir=root_dir,
        split='train',
        normalize=True
    )
    
    # 打印数据集信息
    print(f"数据集大小: {len(dataset)}")
    
    # 获取一个样本
    sample = dataset[0]
    print("\n样本信息:")
    print(f"VFM图像尺寸: {sample['image'].shape}")
    print(f"原始图像尺寸: {sample['original_image'].shape}")
    print(f"掩码尺寸: {sample['mask'].shape}")
    print(f"图像路径: {sample['path']}")
    
    # 测试数据加载器
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    batch = next(iter(dataloader))
    print("\n批次信息:")
    print(f"批次大小: {batch['image'].shape[0]}")
    print(f"VFM图像尺寸: {batch['image'].shape}")
    print(f"原始图像尺寸: {batch['original_image'].shape}")
    print(f"掩码尺寸: {batch['mask'].shape}") 