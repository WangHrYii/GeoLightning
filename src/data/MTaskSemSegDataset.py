import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from lightning import LightningDataModule
import cv2
from osgeo import gdal
from typing import Any, Dict, Optional
import math

# 假设你有一个日志工具，如果没有，可以替换为标准的 `logging` 或直接 `print`
# 这里我们假设 `RankedLogger` 存在且可用
try:
    from src.utils import RankedLogger
    log = RankedLogger(__name__, rank_zero_only=True)
except ImportError:
    import logging
    log = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)


# --- 辅助函数 (保持不变) ---
def read_image_with_gdal(filename):
    """使用GDAL读取图像，返回ndarray，形状为 (bands, height, width)"""
    dataset = gdal.Open(str(filename)) # Pathlib对象转为str
    if not dataset:
        raise FileNotFoundError(f"无法打开文件: {filename}")
    img = dataset.ReadAsArray()
    if img.ndim == 2:
        img = np.expand_dims(img, axis=0)
    return img.astype(np.float32)

def replace_invalid_values(data, nodata_values):
    """将数据中的Nodata值和NaN值替换为0"""
    for nodata in nodata_values:
        if nodata is not None: # 确保nodata值不是None
            data[data == nodata] = 0
    data[np.isnan(data)] = 0
    return data

# --- Dataset 类 (保持不变) ---
class PatchedSSegDataset(Dataset):
    """
    为预先裁剪好的 patches 设计的 Dataset。
    它会同时加载 RGB, TreeMask, 和 nDSM 的 patch。
    """
    def __init__(self, rgb_dir, mask_dir, ndsm_dir, mask_values, transform=None):
        self.rgb_dir = Path(rgb_dir)
        self.mask_dir = Path(mask_dir)
        self.ndsm_dir = Path(ndsm_dir)
        self.mask_values = mask_values
        self.transform = transform

        self.ids = [splitext(file)[0] for file in listdir(self.rgb_dir) if isfile(join(self.rgb_dir, file)) and not file.startswith('.')]
        
        if not self.ids:
            raise RuntimeError(f'No input file found in {self.rgb_dir}')
        log.info(f'Creating dataset for {self.rgb_dir.parent.name} split with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def process_mask(mask_array, mask_values):
        """将掩码值转换为类别索引"""
        h, w = mask_array.shape[1], mask_array.shape[2]
        mask = np.zeros((h, w), dtype=np.int64)
        mask_squeezed = mask_array.squeeze(0)
        for i, v in enumerate(mask_values):
            mask[mask_squeezed == v] = i
        return mask

    def __getitem__(self, idx):
        name = self.ids[idx]
        rgb_file = self.rgb_dir / f'{name}.tif'
        mask_file = self.mask_dir / f'{name}.tif'
        ndsm_file = self.ndsm_dir / f'{name}.tif'

        img_raw = read_image_with_gdal(rgb_file)
        mask_raw = read_image_with_gdal(mask_file)
        ndsm_raw = read_image_with_gdal(ndsm_file)
        
        img = replace_invalid_values(img_raw, [np.nan, -9999])
        mask = self.process_mask(mask_raw, self.mask_values)
        ndsm = replace_invalid_values(ndsm_raw, [np.nan, -9999])
        ndsm = np.squeeze(ndsm, axis=0)
        assist = mask * ndsm
        
        img = img.transpose(1, 2, 0)
        if mask.ndim == 2: mask = np.expand_dims(mask, axis=-1)
        if assist.ndim == 2: assist = np.expand_dims(assist, axis=-1)
        
        assist = assist.astype(np.float32)

        if self.transform:
            transformed = self.transform(image=img, mask=mask, assist=assist)
            img = transformed['image']
            mask = transformed['mask']
            assist = transformed['assist']
        
        if img.shape[0] > 3:
            img = img[:3, :, :]
            
        if mask.ndim == 3: mask = mask.squeeze(-1)  # squeeze(0)的作用是：
        if assist.ndim == 3: assist = assist.squeeze(-1)
        
        assist[assist < 0] = 0
        
        return img, mask, assist, name


# --- 修改后的 LightningDataModule (包含修复) ---
class PatchedSSegDataModule(LightningDataModule):
    def __init__(self, data_dir: str, mask_values=[0,1], batch_size=16, num_workers=8, patch_size=256, model_patch_size=14):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.mask_values = mask_values
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.patch_size = patch_size
        self.model_patch_size = model_patch_size
        self.save_hyperparameters()

        # 核心修复：计算目标尺寸，使其成为模型patch size的整数倍
        # 策略：向上取整到最接近的倍数
        # 例如: 256 / 14 = 18.28 -> ceil -> 19 -> 19 * 14 = 266
        self.target_size = math.ceil(self.patch_size / self.model_patch_size) * self.model_patch_size
        
        if self.target_size != self.patch_size:
            log.warning(f"Input patch size ({self.patch_size}) is not a multiple of model patch size ({self.model_patch_size}).")
            log.warning(f"Automatically resizing all input patches to {self.target_size}x{self.target_size}.")

        # 训练集的完整变换
        self.train_transform = A.Compose([
            # 1. 调整尺寸以满足模型要求
            A.Resize(height=self.target_size, width=self.target_size, interpolation=cv2.INTER_LINEAR),
            # 2. 其他数据增强
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5),
            A.Transpose(p=0.5),
            # 3. 转换为Tensor
            ToTensorV2()
        ], additional_targets={'assist': 'mask'})
        
        # 验证集和测试集的基本变换
        self.val_transform = A.Compose([
            # 1. 同样需要调整尺寸
            A.Resize(height=self.target_size, width=self.target_size, interpolation=cv2.INTER_LINEAR),
            # 2. 转换为Tensor
            ToTensorV2()
        ], additional_targets={'assist': 'mask'})

    def setup(self, stage: Optional[str] = None):
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        log.info("Setting up train dataset...")
        self.data_train = PatchedSSegDataset(
            rgb_dir=self.data_dir / 'train' / 'RGB',
            mask_dir=self.data_dir / 'train' / 'TreeMask',
            ndsm_dir=self.data_dir / 'train' / 'nDSM',
            mask_values=self.mask_values,
            transform=self.train_transform
        )
        
        log.info("Setting up validation dataset...")
        self.data_val = PatchedSSegDataset(
            rgb_dir=self.data_dir / 'val' / 'RGB',
            mask_dir=self.data_dir / 'val' / 'TreeMask',
            ndsm_dir=self.data_dir / 'val' / 'nDSM',
            mask_values=self.mask_values,
            transform=self.val_transform
        )
        
        log.info("Setting up test dataset...")
        self.data_test = PatchedSSegDataset(
            rgb_dir=self.data_dir / 'test' / 'RGB',
            mask_dir=self.data_dir / 'test' / 'TreeMask',
            ndsm_dir=self.data_dir / 'test' / 'nDSM',
            mask_values=self.mask_values,
            transform=self.val_transform
        )
        
        log.info(f"Datasets loaded: {len(self.data_train)} train, {len(self.data_val)} val, {len(self.data_test)} test samples.")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.data_train, batch_size=self.batch_size, num_workers=self.num_workers,
            pin_memory=True, shuffle=True, drop_last=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.data_val, batch_size=self.batch_size, num_workers=self.num_workers,
            pin_memory=True, shuffle=False
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.data_test, batch_size=self.batch_size, num_workers=self.num_workers,
            pin_memory=True, shuffle=False
        )
    
    def teardown(self, stage: Optional[str] = None) -> None:
        pass

    def state_dict(self) -> Dict[Any, Any]:
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        pass


# --- 修改后的继承类，用于传递新参数 ---
class TreeHeightPatchedDataset(PatchedSSegDataModule):
    def __init__(self, data_dir: str, mask_values=[0, 1], batch_size=16, num_workers=8, patch_size=256, model_patch_size=14):
        super().__init__(
            data_dir=data_dir, 
            mask_values=mask_values, 
            batch_size=batch_size, 
            num_workers=num_workers,
            patch_size=patch_size,
            model_patch_size=model_patch_size
        )


# --- Debug 和使用示例 ---
if __name__ == '__main__':
    # --- 配置 ---
    # 1. 修改为你的 patched_data 根目录
    DATA_DIR = "/path/to/your/patched_data_folder/"
    BATCH_SIZE = 4
    
    # 2. 检查路径是否已配置
    if "path/to/your" in DATA_DIR or not os.path.exists(DATA_DIR):
        print("="*80)
        print("!! 错误：请在脚本的 `if __name__ == '__main__':` 部分修改 `DATA_DIR` 为你有效的项目路径 !!")
        print("="*80)
    else:
        print("--- 实例化 DataModule ---")
        # 实例化时传入模型patch size
        # 假设你的数据是256x256，模型是DINOv2 (patch size 14)
        dm = TreeHeightPatchedDataset(
            data_dir=DATA_DIR,
            mask_values=[0, 1],        # 背景=0, 树=1
            batch_size=BATCH_SIZE,
            num_workers=4,             # 根据你的CPU核心数调整
            patch_size=256,            # 你的数据patch大小
            model_patch_size=14        # DINOv2的模型patch大小
        )
        
        print("\n--- 设置并检查数据 ---")
        dm.setup()
        
        print("\n--- 检查 DataLoader 输出 ---")
        train_loader = dm.train_dataloader()
        img_batch, mask_batch, assist_batch, name_batch = next(iter(train_loader))
        
        print("\n--- 批次输出验证 ---")
        print(f"最终送入模型的图像批次形状: {img_batch.shape},  数据类型: {img_batch.dtype}")
        print(f"掩码批次形状:                {mask_batch.shape},   数据类型: {mask_batch.dtype}")
        print(f"辅助任务目标批次形状:        {assist_batch.shape},  数据类型: {assist_batch.dtype}")
        
        # 验证最终的图像尺寸是否正确
        final_height, final_width = img_batch.shape[2], img_batch.shape[3]
        model_p_size = dm.hparams.model_patch_size
        print(f"\n验证尺寸是否为 {model_p_size} 的倍数:")
        print(f"高度: {final_height} % {model_p_size} = {final_height % model_p_size}")
        print(f"宽度: {final_width} % {model_p_size} = {final_width % model_p_size}")
        assert final_height % model_p_size == 0
        assert final_width % model_p_size == 0
        print("验证通过！尺寸正确。")
