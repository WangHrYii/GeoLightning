import os
import rasterio
import numpy as np
from PIL import Image
from tqdm import tqdm
import shutil

def create_directory_structure(base_dir):
    """创建数据集目录结构"""
    splits = ['train', 'test']
    for split in splits:
        os.makedirs(os.path.join(base_dir, split, 'images_png'), exist_ok=True)
        os.makedirs(os.path.join(base_dir, split, 'masks_png'), exist_ok=True)

def split_image(image_path, label_path, output_dir, output_mask_dir, tile_size=1024):
    """将大图像和标签切分成指定大小的tiles"""
    with rasterio.open(image_path) as src_img, rasterio.open(label_path) as src_label:
        # 读取图像数据
        image = src_img.read()
        label = src_label.read()
        
        # 转换为RGB格式（假设是3通道）
        if image.shape[0] == 4:  # 如果是4通道，只取RGB
            image = image[:3]
        
        # 获取图像尺寸
        height, width = image.shape[1], image.shape[2]
        
        # 计算可以切分的tile数量
        n_tiles_h = height // tile_size
        n_tiles_w = width // tile_size
        
        # 切分图像和标签
        for i in range(n_tiles_h):
            for j in range(n_tiles_w):
                # 计算当前tile的坐标
                y_start = i * tile_size
                x_start = j * tile_size
                
                # 提取tile
                tile = image[:, y_start:y_start + tile_size, x_start:x_start + tile_size]
                label_tile = label[:, y_start:y_start + tile_size, x_start:x_start + tile_size]
                
                # 转换为PIL图像
                tile = np.transpose(tile, (1, 2, 0))  # 转换为HWC格式
                tile = tile.astype(np.uint8)
                tile_img = Image.fromarray(tile)
                
                # 转换标签
                label_tile = np.transpose(label_tile, (1, 2, 0))  # 转换为HWC格式
                label_tile = label_tile.astype(np.uint8)
                label_img = Image.fromarray(label_tile)
                
                # 生成文件名
                base_name = os.path.splitext(os.path.basename(image_path))[0].replace('_RGB', '')
                tile_name = f"{base_name}_{i}_{j}.png"
                
                # 保存tile和标签
                tile_img.save(os.path.join(output_dir, tile_name))
                label_img.save(os.path.join(output_mask_dir, tile_name))

def process_potsdam_dataset(input_dir, output_dir):
    """处理Potsdam数据集"""
    # 创建目录结构
    create_directory_structure(output_dir)
    
    # 定义数据集划分
    splits = {
        'train': ['2_10', '2_11', '2_12', '3_10', '3_11', '3_12', '4_10', '4_11', '4_12', 
                 '5_10', '5_11', '5_12', '6_10', '6_11', '6_12', '6_7', '6_8', '6_9', 
                 '7_10', '7_11', '7_12', '7_7', '7_8', '7_9'],
        'test': ['5_15', '6_15', '6_13', '3_13', '4_14', '6_14', '5_14', '2_13', '4_15', 
                '2_14', '5_13', '4_13', '3_14', '7_13']
    }
    
    # 处理每个划分
    for split_name, image_ids in splits.items():
        print(f"处理{split_name}集...")
        for img_id in tqdm(image_ids):
            # 构建文件名
            rgb_file = f"top_potsdam_{img_id}_RGB.tif"
            label_file = f"top_potsdam_{img_id}_label.tif"
            
            input_rgb_path = os.path.join(input_dir,'image', rgb_file)
            input_label_path = os.path.join(input_dir, 'label', label_file)
            output_path = os.path.join(output_dir, split_name, 'images_png')
            output_mask_path = os.path.join(output_dir, split_name, 'masks_png')
            
            if os.path.exists(input_rgb_path) and os.path.exists(input_label_path):
                split_image(input_rgb_path, input_label_path, output_path, output_mask_path)
            else:
                if not os.path.exists(input_rgb_path):
                    print(f"警告：找不到RGB文件 {input_rgb_path}")
                if not os.path.exists(input_label_path):
                    print(f"警告：找不到标签文件 {input_label_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess the Potsdam dataset")
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")
    args = parser.parse_args()
    process_potsdam_dataset(args.input_dir, args.output_dir)
