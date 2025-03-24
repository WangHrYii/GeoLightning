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
import rootutils
import cv2
from osgeo import gdal

from typing import Any, Dict, Optional

rootutils.setup_root('/home/whr/Codes/CLFoundation/CLFoundation/src/train.py', indicator=".project-root", pythonpath=True)

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
    def __init__(self, images_dir: str, mask_dir: str, mask_values: list, assist_dir: str, scale: float = 1.0, mask_suffix: str = '_mask', transform=None):
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

        if self.transform != None:
            transformed = self.transform(image=img, mask=mask, assist=assist)
            img = transformed['image']
            mask = transformed['mask']
            assist = transformed['assist']

        return img.astype(np.float32), mask.astype(np.int64), assist.astype(np.float32), name


class BasicMTaskSSegDatasetLD(LightningDataModule):
    """
    Basic segmentation dataset, from LightningDataModule.
    """
    def __init__(self, images_dir: str, mask_dir: str, assist_dir, mask_values=[0,101], batch_size=1 , scale: float = 1.0, mask_suffix: str = ''):
        super().__init__()
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        self.assist_dir = Path(assist_dir)
        self.mask_values = mask_values
        self.batch_size = batch_size
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix
        self.transform = A.Compose([
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
            A.RandomSizedCrop(min_max_height=(int(scale * 0.8 * 256), int(scale * 1.2 * 256)), height=int(scale * 256), width=int(scale * 256), p=0.5),  # RandomSizedCrop的意思是在min_max_height和min_max_width之间随机裁剪，然后再resize到height和width
            A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
            ToTensorV2()
        ])

    def setup(self, stage: str = None):
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not self.mask_dir.exists():
            raise FileNotFoundError(f"Mask directory not found: {self.mask_dir}")
        if not self.assist_dir.exists():
            raise FileNotFoundError(f"Assist directory not found: {self.assist_dir}")
        
        self.data_train = BasicMTaskSSegDataset(os.path.join(self.images_dir, 'train'), os.path.join(self.mask_dir, 'train'),self.mask_values , os.path.join(self.assist_dir, 'train'), self.scale, self.transform)
        self.data_val = BasicMTaskSSegDataset(os.path.join(self.images_dir, 'val'), os.path.join(self.mask_dir, 'val'), self.mask_values ,os.path.join(self.assist_dir, 'val'), self.scale)
        self.data_test = BasicMTaskSSegDataset(os.path.join(self.images_dir, 'val'), os.path.join(self.mask_dir, 'val'), self.mask_values ,os.path.join(self.assist_dir, 'val'), self.scale)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=0,
            pin_memory=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=0,
            pin_memory=True,
            shuffle=False,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=0,
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
    dataset = BasicMTaskSSegDataset(images_dir='/mnt/data/Tree/TreeHeight/raster_1m_crop_02_256/train',
                                    mask_dir='/mnt/data/Tree/TreeHeight/tree_cover_crop_02_256/train',
                                    mask_values=[0, 101],
                                    assist_dir='/mnt/data/Tree/TreeHeight/nDSM_crop_02_256/train',
                                    scale=0.5)

    img, mask, assist = dataset[0]
    print(img.shape, mask.shape, assist.shape)
