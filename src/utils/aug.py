import cv2
import albumentations as A

import random
import numpy as np

def get_aug_transforms(params=None):
    """
        Get augmentation transforms
    Args:
        params (dict): augmentation parameters
    """
    transforms = []
    if params:
        interpolation=cv2.INTER_LINEAR
        border_mode=cv2.BORDER_REFLECT_101
        if 'aug_p' in params:
            aug_p = params['aug_p']
            
        if params.get('crop') is True:
            transforms.append(A.RandomCrop(params['crop_size'], params['crop_size']))
        if params.get('flip') is True:
            transforms.append(A.Flip(p=aug_p))
        if params.get('transpose') is True:
            transforms.append(A.Transpose(p=aug_p))
        if params.get('scale') is True:
            transforms.append(A.RandomScale(scale_limit=params['scale_limit'], interpolation=interpolation, p=aug_p))
        if params.get('shift_scale_rotate') is True:
            transforms.append(A.ShiftScaleRotate(shift_limit=params['ssr_limits'][0], scale_limit=params['ssr_limits'][1], rotate_limit=params['ssr_limits'][2], 
                                                 interpolation=interpolation, border_mode=border_mode, value=None, mask_value=None, p=aug_p))
        if params.get('optical_distortion') is True:
            transforms.append(A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, interpolation=interpolation, 
                                                  border_mode=border_mode, value=None, mask_value=None, p=aug_p))
        if params.get('grid_distortion') is True:
            transforms.append(A.GridDistortion(num_steps=5, distort_limit=0.3, interpolation=cv2.INTER_LINEAR,
                                               border_mode=cv2.BORDER_REFLECT_101, value=None, mask_value=None, p=aug_p))
        if params.get('elasticTransform') is True:
            transforms.append(A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, interpolation=cv2.INTER_LINEAR,
                                                 border_mode=cv2.BORDER_REFLECT_101, value=None, mask_value=None, approximate=False, p=aug_p))
        if params.get('hsv') is True:
            transforms.append(RandomHueSaturationValue(hue_shift_limit=params['hsv_limits'][0], sat_shift_limit=params['hsv_limits'][1], val_shift_limit=params['hsv_limits'][2], p=aug_p))

    transform = A.Compose(transforms)

    return transform

# To customize enhancements, please follow the standards of albumentations
class RandomHueSaturationValue(A.ImageOnlyTransform):
    def __init__(self, hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, always_apply=False, p=0.5):
        super(RandomHueSaturationValue, self).__init__(always_apply, p)
        self.hue_shift_limit = self._check_limit(hue_shift_limit, 'hue_shift_limit')
        self.sat_shift_limit = self._check_limit(sat_shift_limit, 'sat_shift_limit')
        self.val_shift_limit = self._check_limit(val_shift_limit, 'val_shift_limit')
        self.aug = A.HueSaturationValue(hue_shift_limit=self.hue_shift_limit, 
                                        sat_shift_limit=self.sat_shift_limit, 
                                        val_shift_limit=self.val_shift_limit, 
                                        p=1.0)  # Always apply the internal transform

    def _check_limit(self, limit, name):
        if isinstance(limit, int):
            return [-limit, limit]
        elif isinstance(limit, (tuple, list)) and len(limit) == 2:
            return limit
        else:
            raise ValueError(f"Invalid {name}: {limit}")

    def apply(self, img, **params):
        band_count = img.shape[2]
        band_group = []
        for i in range(0, band_count, 3):
            band_list = [i + ii for ii in range(3)]
            if band_list[2] + 1 > band_count:
                band_list = [ii - (band_list[2] + 1 - band_count) for ii in band_list]
            band_group.append(band_list)
        band_group.reverse()
        img_dst = img.copy()
        for i in range(len(band_group)):
            img_src_image = img[:, :, band_group[i][0]:(band_group[i][2] + 1)]
            img_dst_image = self.aug(image=img_src_image)['image']
            for ii in range(3):
                img_dst[:, :, band_group[i][ii]] = img_dst_image[:, :, ii]
        return img_dst
    
    def get_transform_init_args_names(self):
        return ("hue_shift_limit", "sat_shift_limit", "val_shift_limit")