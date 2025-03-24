import torch
from torch.utils.data import Dataset, DataLoader
from lightning import LightningDataModule
from osgeo import gdal
import numpy as np

import torch
import numpy as np
from osgeo import gdal
from torch.utils.data import Dataset, DataLoader

from typing import Optional, List, Tuple

class RSTiledPredDataset(Dataset):
    def __init__(
        self,
        file_path: str,
        patch_size: int = 256,
        overlap: float = 0.25,
        pad_mode: str = 'reflect'
    ):
        """
        Args:
            file_path: 文件路径
            patch_size: 切片大小
            overlap:  重叠率
            pad_mode: 填充模式, 默认为反射填充, 可选值为: 'constant', 'reflect', 'replicate', 'circular'
                      constant: 常数填充, 意思是填充值为常数, 默认为0
                      reflect: 反射填充, 意思是以边缘为轴, 对称复制
                      replicate: 复制填充, 意思是复制最后一个元素
                      circular: 循环填充, 意思是循环填充
        """
        super().__init__()
        self.original_img, self.geotrans, self.proj = self._read_tif(file_path)
        self.patch_size = patch_size
        self.overlap = overlap
        self.pad_mode = pad_mode
        
        # 预处理
        self.padded_img, self.padding = self._pad_image()
        self.coords = self._generate_coords()
        
    def _read_tif(self, file_path: str) -> Tuple[np.ndarray, list, str]:
        """读取多波段TIFF文件"""
        dataset = gdal.Open(file_path)
        if not dataset:
            raise ValueError(f"无法打开文件: {file_path}")
            
        geotrans = dataset.GetGeoTransform()
        proj = dataset.GetProjection()
        bands = dataset.RasterCount
        
        # 读取为CxHxW格式
        img = np.stack([
            dataset.GetRasterBand(i+1).ReadAsArray() 
            for i in range(bands)
        ], axis=0)
        
        return img, geotrans, proj
    
    def _pad_image(self) -> Tuple[np.ndarray, tuple]:
        """多维度填充处理"""
        _, h, w = self.original_img.shape
        pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        
        padded = np.pad(
            self.original_img,
            ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
            mode=self.pad_mode
        )
        return padded, (pad_top, pad_left)
    
    def _generate_coords(self) -> List[Tuple[int, int]]:
        """生成切片坐标列表"""
        _, h, w = self.padded_img.shape
        stride = int(self.patch_size * (1 - self.overlap))
        coords = []
        
        # 主滑动窗口
        for y in range(0, h - self.patch_size + 1, stride):
            for x in range(0, w - self.patch_size + 1, stride):
                coords.append((y, x))
        
        # 边界补偿
        if (h - self.patch_size) % stride != 0:
            y = h - self.patch_size
            for x in range(0, w - self.patch_size + 1, stride):
                coords.append((y, x))
                
        if (w - self.patch_size) % stride != 0:
            x = w - self.patch_size
            for y in range(0, h - self.patch_size + 1, stride):
                coords.append((y, x))
                
        return coords
    
    def __len__(self) -> int:
        return len(self.coords)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
        y, x = self.coords[idx]   # 为什么是y, x而不是x, y? 因为numpy是先行后列
        patch = self.padded_img[
            :, 
            y:y+self.patch_size, 
            x:x+self.patch_size
        ]
        return torch.from_numpy(patch).float(), (y, x)  # 每一个item返回一个patch和对应的坐标

class RSTiledPredDatasetLD(LightningDataModule):
    def __init__(
        self,
        file_path: str,
        patch_size: int = 256,
        overlap: float = 0.25,
        batch_size: int = 8,
        num_workers: int = 4
    ):
        super().__init__()
        self.file_path = file_path
        self.patch_size = patch_size
        self.overlap = overlap
        self.batch_size = batch_size
        self.num_workers = num_workers
        
    def setup(self, stage: Optional[str] = None):
        self.dataset = RSTiledPredDataset(
            self.file_path,
            self.patch_size,
            self.overlap
        )
        
    def predict_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=self.collate_fn
        )

    @staticmethod
    def collate_fn(batch):
        """自定义批次组装"""
        patches = torch.stack([item[0] for item in batch])
        coords = [item[1] for item in batch]
        return patches, coords

