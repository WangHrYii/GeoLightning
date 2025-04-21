import os
import yaml
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import lightning as L
import data.utils as data_utils
from torchvision import transforms as T
import numpy as np
from registries import DATASET_REGISTRY
from src.utils.aug import get_aug_transforms

class RSBaseSegDataset(Dataset):
    def __init__(self, cfg, aug_transforms, mode='train', **kwargs):
        self.cfg = cfg
        self.mode = mode
        self.aug_transforms = aug_transforms

        self._load_config()  # 加载配置
        self._generate_file_lists()  # 生成文件列表

        self._validate_file_list(self.img_files, f'{mode}_imgfiles')
        self._validate_file_list(self.img_files, f'{mode}_gtfiles')

        normalize = T.Normalize(mean=self.mean_value, std=self.std_value)
        self.transforms = T.Compose([
            T.ToTensor(),
            normalize
        ])

    def _load_config(self):
        if self.mode != 'train' and self.mode != 'val':
            raise ValueError("Invalid mode. Must be 'train', 'val'.")

        self.data_root = getattr(self.cfg, f'{self.mode}_img_path')
        self.img_list_file = self.cfg.get(f'{self.mode}_img_list')
        self.gt_root = getattr(self.cfg, f'{self.mode}_gt_path')
        self.gt_list_file = self.cfg.get(f'{self.mode}_gt_list')
        
        self.img_suffix = self.cfg.img_suffix
        self.gt_suffix = self.cfg.gt_suffix
        self.mean_value = data_utils.load_mean_std_file(self.cfg.mean_file)
        self.std_value = data_utils.load_mean_std_file(self.cfg.std_file)

    def _generate_file_lists(self):
        self.img_files = data_utils.generate_file_list(
            self.data_root, self.img_suffix, list_file=self.img_list_file
        )

        if self.gt_suffix == "LastBand":
            self.gt_files = ['0'] * len(self.img_files)
        else:
            self.gt_files = data_utils.generate_file_list(
                self.gt_root, self.gt_suffix, list_file=self.gt_list_file
            )

    def _validate_file_list(self, file_list, list_name):
        """
        验证文件列表的有效性和非空性。

        参数：
            file_list (list): 要验证的文件路径列表。
            list_name (str): 文件列表的名称，用于错误提示。

        异常：
            ValueError: 如果文件列表为空或包含不存在的文件路径。
        """
        if not file_list:
            raise ValueError(f"文件列表 '{list_name}' 为空，清检查路径")

        invalid_files = [file for file in file_list if not os.path.exists(file)]
        if invalid_files:
            raise ValueError(f"以下文件在列表 '{list_name}' 中不存在: {invalid_files}")
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, index):
        img_name = os.path.basename(self.img_files[index])
        img, gt = self.load_data(index)
        if self.mode == 'train':
            img, gt = self.data_augmentation(img, gt)
        img, gt = self.transform(img, gt)
        return {'image': img, 'label': gt, 'index': index, 'name': img_name}
    
    def get_size(self):
        img, _ = self.load_data(0)
        return [img.shape[0], img.shape[1]]
    
    def get_class_info(self):
        file_path = self.cfg.color_table_file
        return data_utils.parse_class_info_txt(file_path)
    
    def get_color_table(self):
        return self.cfg.color_table_file

    def data_augmentation(self, img, gt):
        transform = get_aug_transforms(self.aug_transforms)
        transformed = transform(image=img, mask=gt)
        image_aug = transformed['image']
        masks_aug = transformed['mask']
        return image_aug, masks_aug
    
    def load_data(self, index):
        img_filename = self.img_files[index]
        gt_filename = self.gt_files[index]
        img = data_utils.gdal_to_numpy(img_filename) # H,W,C
        if self.gt_suffix == "LastBand":
            # if gt_suffix=="LastBand", gt is the last channel of img
            gt = img[:,:,-1]
            img = img[:,:,:-1]
        else:
            gt = data_utils.gdal_to_numpy(gt_filename)[:,:,-1]
        return img, gt

    def transform(self, img, gt):
        img = self.transforms(np.ascontiguousarray(img, dtype = np.uint8)) 
        gt = torch.Tensor(np.ascontiguousarray(gt, dtype = np.float32))
        return img, gt
    
class TestRSBaseSegDataset(Dataset):
    def __init__(self, cfg, aug_transforms):
        self.cfg = cfg
        self.aug_transforms = aug_transforms

        # 获取测试集图像路径
        self.data_root = cfg.test_img_path
        self.img_list_file = cfg.get('test_img_list')
        self.img_suffix = cfg.img_suffix
        self.mean_value = data_utils.load_mean_std_file(cfg.mean_file)
        self.std_value = data_utils.load_mean_std_file(cfg.std_file)
        
        normalize = T.Normalize(mean=self.mean_value, std=self.std_value)
        self.transforms = T.Compose([
            T.ToTensor(),
            normalize
        ])
        # 根据配置生成文件列表
        self.img_filename_list = data_utils.generate_file_list(
            self.data_root, self.img_suffix, list_file=self.img_list_file
        )

        self.imgs = self.img_filename_list

    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, index):
        img = self.load_data(index)
        # 确保数据类型正确，并应用变换
        img = self.transform(img)

        return {'image': img, 'index': index}
    
    def load_data(self, index):
        img_filename = self.imgs[index]
        img = data_utils.gdal_to_numpy(img_filename)
        return img
    
    def transform(self, img):
        img = self.transforms(np.ascontiguousarray(img, dtype = np.uint8)) 
        return img

@DATASET_REGISTRY.register("rs_seg_dataset")
class RSBaseSegDataModule(L.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.aug_transforms = None
        if self.cfg.use_data_enhancement:
            self.aug_transforms = {
                'aug_p': self.cfg.aug_p,
                'crop': self.cfg.aug_crop,
                'crop_size': self.cfg.aug_crop_size,
                'flip': self.cfg.aug_flip, 
                'transpose': self.cfg.aug_transpose,
                'scale': self.cfg.aug_scale,
                'scale_limit': self.cfg.aug_scale_limit,
                'shift_scale_rotate': self.cfg.aug_shift_scale_rotate,
                'ssr_limits': self.cfg.aug_ssr_limits,
                'optical_distortion': self.cfg.aug_optical_distortion,
                'grid_distortion': self.cfg.aug_grid_distortion,
                'elasticTransform': self.cfg.aug_elastic_transform,
                'hsv': self.cfg.aug_hsv,
                'hsv_limits': self.cfg.aug_hsv_limit,
            }
    def prepare_data(self):
        # 在这里可以做一些数据准备工作，例如检查文件是否存在等。
        # 由于你的数据集已经下载好，这里可以留空。
        pass

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:   
            self.train_dataset  =   RSBaseSegDataset(self.cfg, self.aug_transforms, mode='train')
            self.val_dataset    =   RSBaseSegDataset(self.cfg, self.aug_transforms, mode='val')

        if stage == 'predict':
            self.pre_dataset   =   TestRSBaseSegDataset(self.cfg, self.aug_transforms)


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.cfg.get("train_batch_size", 32), shuffle=True, num_workers=self.cfg.get("data_workers", 4))

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.cfg.get("val_batch_size", 32), shuffle=False, num_workers=self.cfg.get("data_workers", 4))

    def predict_dataloader(self):
        return DataLoader(self.pre_dataset, batch_size=self.cfg.get("val_batch_size", 32), shuffle=False, num_workers=self.cfg.get("data_workers", 4))