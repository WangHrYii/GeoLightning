#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAR-Optical语义分割简化推理脚本（处理已分割的切片）
避免复杂的hydra和utils依赖
"""

import torch
import rootutils
import argparse
from lightning import Trainer
from pathlib import Path

# 设置根目录
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.MultiModalSegTask.SAROpticalSeg import SAROpticalSegmentation
from src.data.TiledInferenceDataset import SAROpticalTiledInferenceDataModule
from src.callbacks.TiledInferenceCallback import SAROpticalTiledPredictionCallback


def parse_args():
    parser = argparse.ArgumentParser(description="Run SAR-optical tiled inference")
    parser.add_argument("--sar-dir", required=True)
    parser.add_argument("--optical-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="outputs/segmentation_results")
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--devices", default="1")
    return parser.parse_args()


def main():
    """简化的推理入口函数"""
    args = parse_args()
    sar_dir = args.sar_dir
    optical_dir = args.optical_dir
    ckpt_path = args.checkpoint
    output_dir = args.output_dir
    
    print("=" * 50)
    print("SAR-Optical语义分割推理")
    print("=" * 50)
    print(f"SAR目录: {sar_dir}")
    print(f"光学目录: {optical_dir}")
    print(f"检查点: {ckpt_path}")
    print(f"输出目录: {output_dir}")
    print()
    
    # 验证路径
    if not Path(sar_dir).exists():
        raise FileNotFoundError(f"SAR目录不存在: {sar_dir}")
    if not Path(optical_dir).exists():
        raise FileNotFoundError(f"光学目录不存在: {optical_dir}")
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(f"检查点文件不存在: {ckpt_path}")
    
    # 创建数据模块
    print("初始化数据模块...")
    datamodule = SAROpticalTiledInferenceDataModule(
        sar_dir=sar_dir,
        optical_dir=optical_dir,
        image_size=(512, 512),
        batch_size=8,
        num_workers=4,
        sar_channels=1,
        optical_channels=3,
        sar_normalization="linear",
        optical_normalization=False,
        file_extensions=['.tif', '.tiff', '.png', '.jpg']
    )
    
    # 创建模型
    print("初始化模型...")
    model = SAROpticalSegmentation(
        num_classes=6,
        backbone='ResNet101',
        pretrained=True,
        att_type=None,
        learning_rate=1e-4,
        weight_decay=1e-4,
        optimizer='adamw',
        scheduler='cosine',
        max_epochs=100,
        loss_config={
            'ce_weight': 1.0,
            'dice_weight': 1.0,
            'focal_weight': 0.5
        },
        class_weights=None,
        ignore_index=-100,
        monitor_metric='val_miou'
    )
    
    # 创建回调函数
    print("初始化回调函数...")
    prediction_callback = SAROpticalTiledPredictionCallback(
        output_dir=output_dir,
        num_classes=6,
        save_probability=False,
        save_colored=True,
        color_map=None
    )
    
    # 创建训练器
    print("初始化训练器...")
    trainer = Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        precision="16-mixed",
        logger=False,
        callbacks=[prediction_callback]
    )
    
    # 开始推理
    print("开始推理...")
    try:
        trainer.predict(model, datamodule=datamodule, ckpt_path=ckpt_path)
        print("\n推理完成！")
        print(f"结果已保存到: {output_dir}")
    except Exception as e:
        print(f"推理过程中出现错误: {e}")
        raise


if __name__ == "__main__":
    main()
