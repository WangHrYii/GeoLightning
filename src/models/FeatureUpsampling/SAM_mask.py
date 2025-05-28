from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import torch
import os
import logging
from typing import List, Dict, Any
from PIL import Image
import numpy as np
import torchvision.transforms as T
from lightning import LightningModule

# SAMMaskProcessor是一个工具类
# 它主要用于生成和处理SAM掩码，而不是直接参与训练循环
class SAMMaskProcessor(LightningModule):
    """SAM掩码处理器，负责生成和缓存SAM掩码"""
    
    def __init__(
        self,
        sam_config,
        sam_checkpoint,
        mask_cache_dir: str = "temp_masks",
        experiment_name: str = "",
        vfm_input_size: int = 224,
        points_per_side: int = 32,
        pred_iou_thresh: float = 0.8,
        stability_score_thresh: float = 0.8,
        crop_n_layers: int = 1,
        crop_nms_thresh: float = 0.7,
        crop_overlap_ratio: float = 0.2,
        crop_n_points_downscale_factor: int = 2,
        min_mask_region_area: int = 100
    ):
        """
        初始化SAM掩码处理器
        
        参数:
            sam_config: SAM2配置
            sam_checkpoint: SAM2检查点
            mask_cache_dir: 掩码缓存目录
            experiment_name: 实验名称，用于区分不同实验的缓存
            vfm_input_size: VFM输入尺寸
            points_per_side: 每边的点数
            pred_iou_thresh: 预测IoU阈值
            stability_score_thresh: 稳定性分数阈值
            crop_n_layers: 裁剪层数
            crop_nms_thresh: 裁剪NMS阈值
            crop_overlap_ratio: 裁剪重叠比例
            crop_n_points_downscale_factor: 裁剪点数下采样因子
            min_mask_region_area: 最小掩码区域面积
        """
        super().__init__()
        self.sam_model = build_sam2(sam_config, sam_checkpoint, apply_postprocessing=False)
        self.mask_generator = SAM2AutomaticMaskGenerator(
            model=self.sam_model,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            crop_n_layers=crop_n_layers,
            crop_nms_thresh=crop_nms_thresh,
            crop_overlap_ratio=crop_overlap_ratio,
            crop_n_points_downscale_factor=crop_n_points_downscale_factor,   
            min_mask_region_area=min_mask_region_area
        )

        self.experiment_name = experiment_name
        # 使用实验名称创建独立的缓存目录，避免不同实验间缓存冲突
        if experiment_name:
            self.mask_cache_dir = f"{mask_cache_dir}_{experiment_name}"
        else:
            self.mask_cache_dir = mask_cache_dir
        os.makedirs(self.mask_cache_dir, exist_ok=True)
        
        # 用于生成缩放后的SAM掩码的转换
        self.resize_transform = T.Resize(
            (vfm_input_size, vfm_input_size), 
            interpolation=T.InterpolationMode.NEAREST
        )

    
    def get_sam_masks(
        self, 
        original_img: torch.Tensor, 
        img_path: str
    ) -> List[Dict[str, Any]]:
        """
        获取SAM掩码，优先从缓存读取，不存在则生成新掩码
        
        参数:
            original_img: 原始高分辨率图像 [C, H, W]
            img_path: 图像路径，用于生成缓存路径
            
        返回:
            masks_info: 缩放到VFM输入尺寸的掩码信息列表，每个元素包含掩码张量和面积
        """
        # 生成缓存文件名
        img_name = os.path.basename(img_path)
        cache_name = f"{os.path.splitext(img_name)[0]}_masks.pt"
        if self.experiment_name:
            cache_name = f"{self.experiment_name}_{cache_name}"
            
        cache_path = os.path.join(self.mask_cache_dir, cache_name)
        
        # 检查是否存在缓存
        if os.path.exists(cache_path):
            try:
                return torch.load(cache_path, map_location=original_img.device)
            except Exception as e:
                logging.warning(f"加载缓存掩码失败: {e}，重新生成掩码")
        
        # 如果不存在缓存或加载失败，则生成新掩码
        # 转换为numpy数组供SAM处理
        img_np = original_img.cpu().numpy().transpose(1, 2, 0)
        
        # 生成掩码
        try:
            masks = self.mask_generator.generate(img_np)
            
            # 检查masks是否为空
            if not masks or len(masks) == 0:
                logging.warning(f"SAM未能生成有效掩码: {img_path}")
                return []
            
            # 创建掩码信息列表
            masks_info = []
            for mask in masks:
                if 'segmentation' in mask:
                    # 获取掩码并转换为张量
                    m = torch.from_numpy(mask['segmentation']).float()
                    area = mask['area']
                    
                    # 调整掩码尺寸以匹配VFM输入
                    m_pil = Image.fromarray(m.numpy().astype(np.uint8) * 255)
                    m_resized = self.resize_transform(m_pil)
                    m_tensor = torch.from_numpy(np.array(m_resized)).float() / 255.0
                    
                    masks_info.append({
                        'mask': m_tensor,
                        'area': area
                    })
            
            # 按掩码面积从大到小排序
            masks_info = sorted(masks_info, key=lambda x: x['area'], reverse=True)
            
            # 保存到缓存
            torch.save(masks_info, cache_path)
            
            return masks_info
            
        except Exception as e:
            logging.error(f"SAM生成掩码时出错: {e}")
            return []
    
    def get_original_sam_masks(
        self, 
        original_img: torch.Tensor, 
        img_path: str
    ) -> List[Dict[str, Any]]:
        """
        获取原始SAM掩码，返回完整的掩码信息（包含segmentation字段）
        
        参数:
            original_img: 原始高分辨率图像 [C, H, W]
            img_path: 图像路径，用于生成缓存路径
            
        返回:
            masks: 原始掩码信息列表
        """
        # 生成缓存文件名
        img_name = os.path.basename(img_path)
        cache_name = f"{os.path.splitext(img_name)[0]}_original_masks.pt"
        if self.experiment_name:
            cache_name = f"{self.experiment_name}_{cache_name}"
            
        cache_path = os.path.join(self.mask_cache_dir, cache_name)
        
        # 检查是否存在缓存
        if os.path.exists(cache_path):
            try:
                return torch.load(cache_path, map_location=original_img.device)
            except Exception as e:
                logging.warning(f"加载原始缓存掩码失败: {e}，重新生成掩码")
        
        # 转换为numpy数组供SAM处理
        img_np = original_img.cpu().numpy().transpose(1, 2, 0)
        
        # 生成SAM掩码
        try:
            masks = self.mask_generator.generate(img_np)
            
            # 检查masks是否为空
            if not masks or len(masks) == 0:
                logging.warning(f"SAM未能生成有效掩码: {img_path}")
                return []
            
            # 按掩码面积从大到小排序
            sorted_masks = sorted(masks, key=lambda x: x['area'], reverse=True)
            
            # 保存到缓存
            torch.save(sorted_masks, cache_path)
            
            return sorted_masks
        except Exception as e:
            logging.error(f"SAM生成原始掩码时出错: {e}")
            return []   