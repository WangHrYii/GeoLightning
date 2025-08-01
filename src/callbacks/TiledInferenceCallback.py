#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAR-Optical语义分割推理回调函数（处理已分割的切片）
处理预测结果并保存每个切片的分割结果
"""

import torch
import numpy as np
from osgeo import gdal
import lightning as L
from typing import List, Tuple, Any, Optional
from pathlib import Path


class SAROpticalTiledPredictionCallback(L.Callback):
    """
    SAR-Optical语义分割预测回调函数（处理已分割的切片）
    处理切片预测结果并保存每个切片的分割图像
    """
    
    def __init__(
        self,
        output_dir: str = "segmentation_results",
        num_classes: int = 6,
        save_probability: bool = False,
        save_colored: bool = True,
        color_map: Optional[List[Tuple[int, int, int]]] = None
    ):
        """
        初始化回调函数
        
        Args:
            output_dir: 输出目录
            num_classes: 类别数量
            save_probability: 是否保存概率图
            save_colored: 是否保存彩色分割图
            color_map: 类别颜色映射，如果为None则使用默认颜色
        """
        super().__init__()
        self.output_dir = Path(output_dir)
        self.num_classes = num_classes
        self.save_probability = save_probability
        self.save_colored = save_colored
        
        # 设置默认颜色映射
        if color_map is None:
            self.color_map = [
                (0, 0, 0),       # 类别0：黑色
                (255, 0, 0),     # 类别1：红色
                (0, 255, 0),     # 类别2：绿色
                (0, 0, 255),     # 类别3：蓝色
                (255, 255, 0),   # 类别4：黄色
                (255, 0, 255),   # 类别5：紫色
                (0, 255, 255),   # 类别6：青色
                (128, 128, 128), # 类别7：灰色
            ]
        else:
            self.color_map = color_map
        
        # 确保颜色数量足够
        while len(self.color_map) < num_classes:
            # 生成随机颜色
            import random
            self.color_map.append((
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255)
            ))
        
        # 统计信息
        self.processed_count = 0
        self.class_statistics = {i: 0 for i in range(num_classes)}
    
    def on_predict_start(self, trainer, pl_module):
        """预测开始时的处理"""
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建子目录
        (self.output_dir / "labels").mkdir(exist_ok=True)
        if self.save_probability:
            (self.output_dir / "probabilities").mkdir(exist_ok=True)
        if self.save_colored:
            (self.output_dir / "colored").mkdir(exist_ok=True)
        
        self.processed_count = 0
        self.class_statistics = {i: 0 for i in range(self.num_classes)}
        
        print(f"开始SAR-Optical语义分割预测...")
        print(f"输出目录: {self.output_dir}")
    
    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """每个预测批次结束时的处理"""
        predictions = outputs
        
        # 将预测结果转换为numpy数组
        if isinstance(predictions, torch.Tensor):
            pred_np = predictions.cpu().numpy()  # [batch, num_classes, H, W]
        else:
            pred_np = predictions
        
        batch_size = pred_np.shape[0]
        
        # 处理批次中的每个样本
        for i in range(batch_size):
            # 获取样本信息
            basename = batch['basename'][i]
            sar_path = batch['sar_path'][i]
            optical_path = batch['optical_path'][i]
            
            # 获取单个样本的预测结果 [num_classes, H, W]
            single_pred = pred_np[i]
            
            # 生成分割结果
            segmentation = np.argmax(single_pred, axis=0).astype(np.uint8)  # [H, W]
            
            # 更新统计信息
            unique_classes, counts = np.unique(segmentation, return_counts=True)
            for class_id, count in zip(unique_classes, counts):
                if class_id < self.num_classes:
                    self.class_statistics[class_id] += count
            
            # 保存结果
            self._save_tile_result(
                basename=basename,
                segmentation=segmentation,
                probabilities=single_pred if self.save_probability else None,
                sar_path=sar_path,
                optical_path=optical_path
            )
            
            self.processed_count += 1
        
        if batch_idx % 10 == 0:
            print(f"已处理批次: {batch_idx + 1}, 累计切片: {self.processed_count}")
    
    def on_predict_epoch_end(self, trainer, pl_module):
        """预测轮次结束时的处理"""
        print(f"\n预测完成！")
        print(f"总计处理切片: {self.processed_count}")
        print(f"结果保存在: {self.output_dir}")
        
        # 打印统计信息
        self._print_statistics()
        
        # 保存统计信息到文件
        self._save_statistics()
    
    def _save_tile_result(
        self,
        basename: str,
        segmentation: np.ndarray,
        probabilities: Optional[np.ndarray],
        sar_path: str,
        optical_path: str
    ):
        """保存单个切片的结果"""
        h, w = segmentation.shape
        
        # 1. 保存分割标签图
        label_path = self.output_dir / "labels" / f"{basename}.tif"
        self._save_geotiff(
            array=segmentation[..., np.newaxis],  # [H, W, 1]
            filepath=label_path,
            dtype=gdal.GDT_Byte
        )
        
        # 2. 保存概率图（如果需要）
        if self.save_probability and probabilities is not None:
            prob_path = self.output_dir / "probabilities" / f"{basename}_prob.tif"
            # 转换为HWC格式
            prob_hwc = probabilities.transpose(1, 2, 0)  # [H, W, num_classes]
            self._save_geotiff(
                array=prob_hwc,
                filepath=prob_path,
                dtype=gdal.GDT_Float32
            )
        
        # 3. 保存彩色分割图（如果需要）
        if self.save_colored:
            colored_path = self.output_dir / "colored" / f"{basename}_colored.tif"
            colored_img = self._create_colored_image(segmentation)
            self._save_geotiff(
                array=colored_img,
                filepath=colored_path,
                dtype=gdal.GDT_Byte
            )
    
    def _create_colored_image(self, segmentation: np.ndarray) -> np.ndarray:
        """创建彩色分割图"""
        h, w = segmentation.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        
        for class_id in range(min(self.num_classes, len(self.color_map))):
            mask = segmentation == class_id
            colored[mask] = self.color_map[class_id]
        
        return colored
    
    def _save_geotiff(
        self,
        array: np.ndarray,
        filepath: Path,
        dtype=gdal.GDT_Byte
    ):
        """保存GeoTIFF文件（不包含地理信息）"""
        driver = gdal.GetDriverByName('GTiff')
        
        if array.ndim == 2:
            h, w = array.shape
            bands = 1
            array = array[..., np.newaxis]
        else:
            h, w, bands = array.shape
        
        # 创建数据集
        dataset = driver.Create(
            str(filepath),
            w, h, bands,
            dtype,
            options=['TILED=YES', 'COMPRESS=LZW']
        )
        
        # 写入数据
        for i in range(bands):
            band = dataset.GetRasterBand(i + 1)
            band.WriteArray(array[:, :, i])
            band.FlushCache()
        
        dataset.FlushCache()
        dataset = None
    
    def _print_statistics(self):
        """打印分割统计信息"""
        print("\n=== 分割结果统计 ===")
        total_pixels = sum(self.class_statistics.values())
        
        if total_pixels > 0:
            for class_id in range(self.num_classes):
                count = self.class_statistics[class_id]
                percentage = count / total_pixels * 100
                print(f"类别 {class_id}: {count:,} 像素 ({percentage:.2f}%)")
            
            print(f"总像素数: {total_pixels:,}")
        else:
            print("未找到有效像素")
        
        print("=" * 25)
    
    def _save_statistics(self):
        """保存统计信息到文件"""
        stats_file = self.output_dir / "statistics.txt"
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write("SAR-Optical语义分割结果统计\n")
            f.write("=" * 40 + "\n")
            f.write(f"处理切片数量: {self.processed_count}\n")
            f.write(f"类别数量: {self.num_classes}\n\n")
            
            total_pixels = sum(self.class_statistics.values())
            f.write("类别像素统计:\n")
            
            if total_pixels > 0:
                for class_id in range(self.num_classes):
                    count = self.class_statistics[class_id]
                    percentage = count / total_pixels * 100
                    f.write(f"类别 {class_id}: {count:,} 像素 ({percentage:.2f}%)\n")
                
                f.write(f"\n总像素数: {total_pixels:,}\n")
            else:
                f.write("未找到有效像素\n")
        
        print(f"统计信息已保存到: {stats_file}")
