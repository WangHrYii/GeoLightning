import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset, random_split
from lightning import LightningDataModule
import rootutils
import cv2
from osgeo import gdal

from typing import Any, Dict, Optional

rootutils.setup_root('/home/whr/Codes/GeoLightning/src/train.py', indicator=".project-root", pythonpath=True)

from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def read_image_with_gdal(filename):
    """
    使用GDAL读取图像，返回ndarray
    :param filename: 输入影像路径
    :return: ndarray格式的影像数据，形状为 (bands, height, width)
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

    return img


def replace_invalid_values(data, nodata_values):
    """
    将数据中的Nodata值和NaN值替换为0
    """
    for nodata in nodata_values:
        data[data == nodata] = 0
    data[np.isnan(data)] = 0
    return data


class BasicMTaskSSegDataset(Dataset):
    def __init__(self, images_dir, mask_dir, mask_values, assist_dir, scale: float = 1.0, mask_suffix: str = '_mask', transform=None):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        self.mask_values = mask_values
        self.assist_dir = Path(assist_dir)
        self.transform = transform

        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale

        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]

        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        log.info(f'Creating dataset with {len(self.ids)} examples')  # log.info()会自动只在rank=0的进程上打印日志

    def __len__(self):
        return len(self.ids)


    @staticmethod
    def preprocess(mask_values, img_array, scale, is_mask):
        img = img_array
        c, w, h =img.shape[0], img.shape[2], img.shape[1]  # C, H, W
        newW, newH = int(scale * w), int(scale * h)

        # 需要resize到最接近14的倍数的大小
        newW = int(np.ceil(newW / 14) * 14)
        newH = int(np.ceil(newH / 14) * 14)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        
        # 一个支持多波段的resize
        if c==1:
            img = cv2.resize(img.transpose(1, 2, 0), (newW, newH), interpolation=cv2.INTER_LINEAR)
        else:
            img = cv2.resize(img.transpose(1, 2, 0), (newW, newH), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1)     

        if is_mask:
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i

            return mask

        else:
            img = replace_invalid_values(img, [np.nan, -9999])
            if img.ndim == 2:
                img = img[np.newaxis, ...]

            return img

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(name + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))
        assist_file = list(self.assist_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        assert len(assist_file) == 1, f'Either no assist or multiple assists found for the ID {name}: {assist_file}'

        mask = read_image_with_gdal(mask_file[0])
        img = read_image_with_gdal(img_file[0])
        assist = read_image_with_gdal(assist_file[0])

        img    = self.preprocess(self.mask_values, img, self.scale, is_mask=False)
        mask   = self.preprocess(self.mask_values, mask, self.scale, is_mask=True)
        assist = self.preprocess(self.mask_values, assist, self.scale, is_mask=False)
        # assist ndarray去掉无效维度
        assist = np.squeeze(assist, axis=0)
        assist = mask * assist  # 对应位置相乘

        # 确保所有输入的形状一致
        if img.ndim == 3:
            img = img.transpose(1, 2, 0)  # 转换为 (H, W, C) 格式
        if mask.ndim == 2:
            mask = np.expand_dims(mask, axis=-1)  # 转换为 (H, W, 1) 格式
        if assist.ndim == 2:
            assist = np.expand_dims(assist, axis=-1)  # 转换为 (H, W, 1) 格式

        assist = assist.astype(np.float32)
        if self.transform != None:
            transformed = self.transform(image=img, mask=mask, assist=assist)
            img = transformed['image']
            mask = transformed['mask']
            assist = transformed['assist']
            # img 只取前3个波段
            img = img[:3, :, :]  # H W C
        else:
            transformed = A.Compose([
                ToTensorV2()
            ], additional_targets={'assist': 'mask'})
            transformed = transformed(image=img, mask=mask, assist=assist)
            img = transformed['image']
            mask = transformed['mask']
            assist = transformed['assist']
            # img 只取前3个波段
            img = img[:3, :, :]  # H W C
        # 转换回原始格式
        mask = np.squeeze(mask, axis=-1)  # 转换回 (H, W) 格式
        assist = np.squeeze(assist, axis=-1)  # 转换回 (H, W) 格式

        assist[assist < 0] = 0

        

        # # 可视化img mask assist
        # img = img.numpy().transpose(1, 2, 0)
        # mask = mask.numpy()
        # assist = assist.numpy()
        
        # # mask和assist扩展一个维度
        # mask = np.expand_dims(mask, axis=-1)
        # assist = np.expand_dims(assist, axis=-1)

        # mask = mask*255
        
        # cv2.imwrite('mask.png', mask)
        # cv2.imwrite('assist.png', assist)
        # cv2.imwrite('img.png', img)
        
        return img, mask, assist, name


class BasicMTaskSSegDatasetLD(LightningDataModule):
    """
    Basic segmentation dataset, from LightningDataModule.
    """
    def __init__(self, images_dir: str, mask_dir: str, assist_dir: str, mask_values=[0,1], batch_size=1 , scale: float = 1.0, mask_suffix: str = ''):
        super().__init__()
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        self.assist_dir = Path(assist_dir)
        self.mask_values = mask_values
        self.batch_size = batch_size
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        # 获取一个样本图像来确定尺寸
        sample_ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]
        if sample_ids:
            sample_img = read_image_with_gdal(join(images_dir, sample_ids[0] + '.tif'))
            self.img_height, self.img_width = sample_img.shape[1], sample_img.shape[2]
            # 确保尺寸是14的倍数
            self.img_height = int(np.ceil(self.img_height / 14) * 14)
            self.img_width = int(np.ceil(self.img_width / 14) * 14)
        else:
            raise RuntimeError(f'No input file found in {images_dir}')

        # 训练集的完整变换（包括数据增强）
        self.transform = A.Compose([
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5),
            A.Transpose(p=0.5),
            # A.OneOf([
            #     A.GaussNoise(p=0.5),
            #     A.MultiplicativeNoise(p=0.5)
            # ], p=0.2),
            # A.OneOf([
            #     A.MotionBlur(p=0.2),
            #     A.MedianBlur(blur_limit=3, p=0.1),
            #     A.Blur(blur_limit=3, p=0.1),
            # ], p=0.2),
            # A.OneOf([
            #     A.RandomBrightnessContrast(p=1),
            #     A.RandomGamma(p=1)
            # ], p=0.3),
            # 使用实际图像尺寸进行裁剪，确保裁剪区域在图像范围内
            A.RandomSizedCrop(
                min_max_height=(int(self.img_height * 0.8), self.img_height),
                height=self.img_height,
                width=self.img_width,
                p=0.5,
                interpolation=cv2.INTER_LINEAR
            ),
            # A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
            ToTensorV2()
        ], additional_targets={'assist': 'mask'})
        
        # 验证集和测试集的基本变换（只包含ToTensorV2，不包含数据增强）
        self.val_transform = A.Compose([
            ToTensorV2()
        ], additional_targets={'assist': 'mask'})

    def setup(self, stage: str = None):
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not self.mask_dir.exists():
            raise FileNotFoundError(f"Mask directory not found: {self.mask_dir}")
        if not self.assist_dir.exists():
            raise FileNotFoundError(f"Assist directory not found: {self.assist_dir}")
        
        # 创建完整数据集
        full_dataset = BasicMTaskSSegDataset(images_dir=self.images_dir, 
                                             mask_dir=self.mask_dir, 
                                             assist_dir=self.assist_dir, 
                                             mask_values=self.mask_values, 
                                             scale=self.scale, 
                                             transform=None)  # 不在此应用transform
        
        # 计算训练集和验证集的大小（按8:2拆分）
        dataset_size = len(full_dataset)
        train_size = int(dataset_size * 0.8)
        val_size = dataset_size - train_size
        
        # 使用random_split进行拆分
        train_dataset, val_dataset = random_split(
            full_dataset, 
            [train_size, val_size], 
            generator=torch.Generator().manual_seed(42)  # 固定种子确保可重复性
        )
        
        # 创建带有变换的训练数据集
        class TransformedSubset(Dataset):
            def __init__(self, subset, transform=None):
                self.subset = subset
                self.transform = transform
                
            def __len__(self):
                return len(self.subset)
                
            def __getitem__(self, idx):
                img, mask, assist, name = self.subset[idx]
                
                # 确保输入格式正确
                if isinstance(img, torch.Tensor):
                    img = img.numpy()
                if isinstance(mask, torch.Tensor):
                    mask = mask.numpy()
                if isinstance(assist, torch.Tensor):
                    assist = assist.numpy()
                
                # 应用变换
                if self.transform:
                    # 确保输入格式符合albumentations要求
                    if img.ndim == 3 and img.shape[0] == 3:  # CHW -> HWC
                        img = img.transpose(1, 2, 0)
                    if mask.ndim == 2:
                        mask = np.expand_dims(mask, axis=-1)
                    if assist.ndim == 2:
                        assist = np.expand_dims(assist, axis=-1)
                    
                    transformed = self.transform(image=img, mask=mask, assist=assist)
                    img = transformed['image']
                    mask = transformed['mask']
                    assist = transformed['assist']
                    
                    mask = mask.squeeze(axis=-1)
                    assist = assist.squeeze(axis=-1)
                else:
                    # 使用val_transform
                    transformed = self.val_transform(image=img, mask=mask, assist=assist)
                    img = transformed['image']
                    mask = transformed['mask']
                    assist = transformed['assist']
                
                return img, mask, assist, name
                    
        
        # 应用适当的变换创建最终数据集
        self.data_train = TransformedSubset(train_dataset, self.transform)
        self.data_val = TransformedSubset(val_dataset, self.val_transform)  # 验证集不需要数据增强
        self.data_test = self.data_val                        # 测试集使用相同的验证集数据
        
        log.info(f'Dataset split: {train_size} training samples, {val_size} validation samples')

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=16,
            pin_memory=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=16,
            pin_memory=True,
            shuffle=False,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=16,
            pin_memory=True,
            shuffle=False,
        )
    
    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


class TreeHeightDataset(BasicMTaskSSegDatasetLD):
    def __init__(self, images_dir: str, mask_dir: str, assist_dir: str, scale: float = 1.0, mask_values=[0, 101], batch_size=1):
        super().__init__(images_dir, mask_dir, assist_dir, scale=scale, mask_values=mask_values, batch_size=batch_size)


# debug
if __name__ == '__main__':

    # 测试BasicMTaskSSegDataset

    transform = A.Compose([
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5),
            A.Transpose(p=0.5),
            A.OneOf([
                A.GaussNoise(p=0.5),
                A.MultiplicativeNoise(p=0.5)
            ], p=0.2),
            A.OneOf([
                A.MotionBlur(p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.2),
            A.OneOf([
                A.RandomBrightnessContrast(p=1),
                A.RandomGamma(p=1)
            ], p=0.3),
            # A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
            ToTensorV2()
        ], additional_targets={'assist': 'mask'})

    dataset = BasicMTaskSSegDataset(images_dir='/mnt/data/TreeHeight/image_512',
                                    mask_dir='/mnt/data/TreeHeight/treecover_512',
                                    mask_values=[0, 1],
                                    assist_dir='/mnt/data/TreeHeight/nDSM_512',
                                    scale=1, transform=transform)

    for i in range(len(dataset)):
        img, mask, assist, _ = dataset[i]
        print(img.shape, mask.shape, assist.shape)
