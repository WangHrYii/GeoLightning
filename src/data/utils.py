import os
import numpy as np
from osgeo import gdal
gdal.UseExceptions()
import random

def generate_file_list(file_root, suffix, replace=None, list_file=None):
    """
    生成指定目录下符合条件的文件列表。

    Args:
        file_root (str): 要搜索的目录的路径。
        suffix (str): 要匹配的文件后缀名 (例如: ".txt", ".jpg")。
        replace (list, optional): 一个包含两个字符串的列表，用于替换文件名中的字符。
            第一个元素是要被替换的字符，第二个元素是替换后的字符。
            如果为 None 或空列表，则不进行替换。 Defaults to None.
        list_file (str, optional): 包含文件名列表的文件路径。如果提供此参数，
            函数将直接从该文件中读取文件名列表，而不会遍历 file_root 目录。
            Defaults to None.

    Returns:
        list: 一个包含完整文件路径的列表。
    """

    if list_file:
        # 如果提供了 list_file, 直接从文件中读取
        try:
            with open(list_file, 'r') as f:
                filename_list = [line.strip() for line in f]
                #可在此处进行replace操作
                if replace:
                    filename_list = [name.replace(replace[0],replace[1]) for name in filename_list]
                filename_list.sort()  # 排序
                return filename_list
        except FileNotFoundError:
            print(f"Warning: File not found: {list_file}")
            return []

    filename_list = []
    try:  # 捕获可能的 OSError
        for filename in os.listdir(file_root):
            if filename.endswith(suffix):
                basename = filename[:-len(suffix)]  # 去除后缀
                if replace:
                    basename = basename.replace(replace[0], replace[1])
                full_path = os.path.join(file_root, basename + suffix) # 确保使用 os.path.join
                filename_list.append(full_path)
    except OSError as e:
        print(f"Error accessing directory {file_root}: {e}")
        return []  # 或者 raise, 取决于你希望如何处理错误

    filename_list.sort()  # 排序
    return filename_list

def load_mean_std_file(filepath):
    """
    从文本文件中加载均值或标准差数据。

    Args:
        filepath (str): 包含均值或标准差数据的文件路径。
            文件应每行包含一个浮点数，表示一个通道的均值或标准差。

    Returns:
        list: 包含均值或标准差数据的列表，数据类型为 float。
              如果文件不存在或读取失败，则返回一个空列表。
    """
    try:
        with open(filepath, 'r') as f:
            # 1. 使用列表推导式简化代码，并去除 strip 的重复调用
            values = [float(line.strip()) for line in f]
            return values
    except FileNotFoundError:
        print(f"Warning: File not found: {filepath}")
        return []
    except (ValueError, IOError) as e: # 处理可能的 ValueError (数据类型错误) 和 IOError (其他文件读取错误)
        print(f"Warning: Error reading file {filepath}: {e}")
        return []
    
def gdal_to_numpy(filename):
    """
    使用 GDAL 读取栅格图像，并将其转换为 NumPy 数组。

    Args:
        filename (str): 栅格图像文件的路径。

    Returns:
        numpy.ndarray:  一个 NumPy 数组，表示图像数据。  形状为 (H, W, C)，
                        其中 H 是图像高度，W 是图像宽度，C 是图像通道数。
                        如果无法打开文件，则返回 None。
    """
    dataset = gdal.Open(filename)
    if dataset is None:
        print('WARNING', f'GDAL can not open {filename} !')
        return None

    img_width = dataset.RasterXSize
    img_height = dataset.RasterYSize
    img_nbands = dataset.RasterCount

    # 1. 使用 ReadAsArray 一次读取所有波段，避免循环
    img = dataset.ReadAsArray()  # (Band, Height, Width)

    # 2. 调整维度顺序为 (Height, Width, Band)，并处理单波段图像
    if img.ndim == 2:  # 单波段
        img = img[:, :, np.newaxis]  # (H, W) -> (H, W, 1)
    else:  # 多波段
        img = img.transpose((1, 2, 0))  # (Band, Height, Width) -> (Height, Width, Band)

    return img


import re

def parse_class_info_txt(file_path):
    """
    解析包含类别信息的 TXT 文件。

    Args:
        file_path (str): TXT 文件的路径。

    Returns:
        list: 类别信息列表。每个元素是一个字典，包含 'id', 'name', 'color' 键。
              如果文件读取或解析失败，返回 None。
    """
    class_info = []
    try:
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()  # 去除首尾空白字符
                if not line:  # 跳过空行
                    continue

                # 使用正则表达式匹配
                match = re.match(r"(\d+)/(\d+)/(\d+)#(\d+)_(.+)", line)
                if match:
                    r, g, b, class_id, class_name = match.groups()
                    color = (int(r), int(g), int(b))
                    class_id = int(class_id)
                    class_info.append({
                        "id": class_id,
                        "name": class_name,
                        "color": color,
                    })
                else:
                    print(f"Warning: Skipping invalid line: {line}")

    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return None
    except Exception as e:
        print(f"Error reading or parsing file: {file_path}\n{e}")
        return None
    # 按id排序
    class_info.sort(key=lambda x: x["id"])
    return class_info


