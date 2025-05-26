import os
import numpy as np
from osgeo import gdal
from pathlib import Path
import shutil

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

def is_black_image(img_path, threshold=0.1):
    """
    检查图像是否全黑
    :param img_path: 图像路径
    :param threshold: 判断为黑色的阈值，默认为0.1
    :return: 是否为全黑图像
    """
    try:
        img = read_image_with_gdal(img_path)
        # 计算所有波段的平均值
        mean_value = np.mean(img)
        return mean_value < threshold
    except Exception as e:
        print(f"处理文件 {img_path} 时出错: {str(e)}")
        return False

def main():
    # 设置路径
    base_dir = "/mnt/data/TreeHeight"
    image_dir = os.path.join(base_dir, "image_256")
    ndsm_dir = os.path.join(base_dir, "nDSM_256")
    tree_cover_dir = os.path.join(base_dir, "treecover_256")

    # 创建备份目录
    backup_dir = os.path.join(base_dir, "backup_black_images")
    os.makedirs(backup_dir, exist_ok=True)
    os.makedirs(os.path.join(backup_dir, "image_256"), exist_ok=True)
    os.makedirs(os.path.join(backup_dir, "nDSM_256"), exist_ok=True)
    os.makedirs(os.path.join(backup_dir, "treecover_256"), exist_ok=True)

    # 获取所有图像文件
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.tif')]
    total_files = len(image_files)
    black_files = []

    print(f"开始检查 {total_files} 个文件...")

    # 检查每个图像
    for i, img_file in enumerate(image_files, 1):
        img_path = os.path.join(image_dir, img_file)
        if is_black_image(img_path):
            black_files.append(img_file)
            print(f"[{i}/{total_files}] 发现全黑图像: {img_file}")

    # 处理全黑图像及其对应文件
    if black_files:
        print(f"\n找到 {len(black_files)} 个全黑图像，开始移动文件...")
        
        for img_file in black_files:
            # 构建文件路径
            img_path = os.path.join(image_dir, img_file)
            ndsm_path = os.path.join(ndsm_dir, img_file)
            tree_cover_path = os.path.join(tree_cover_dir, img_file)

            # 移动文件到备份目录
            try:
                # 移动图像
                shutil.move(img_path, os.path.join(backup_dir, "image_512", img_file))
                print(f"已移动图像: {img_file}")

                # 移动nDSM
                if os.path.exists(ndsm_path):
                    shutil.move(ndsm_path, os.path.join(backup_dir, "nDSM_512", img_file))
                    print(f"已移动nDSM: {img_file}")

                # 移动tree_cover
                if os.path.exists(tree_cover_path):
                    shutil.move(tree_cover_path, os.path.join(backup_dir, "treecover_512", img_file))
                    print(f"已移动tree_cover: {img_file}")

            except Exception as e:
                print(f"移动文件 {img_file} 时出错: {str(e)}")

        print(f"\n处理完成！")
        print(f"共发现并移动了 {len(black_files)} 个全黑图像及其对应文件")
        print(f"文件已备份到: {backup_dir}")
    else:
        print("\n未发现全黑图像")

if __name__ == "__main__":
    main()
