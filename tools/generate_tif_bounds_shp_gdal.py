#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为文件夹中的所有.tif文件生成矩形边界shapefile并合并 (使用GDAL)
功能：遍历指定文件夹，读取所有.tif文件的地理边界，生成矩形shapefile，并合并到一个文件中
作者：Assistant
日期：2024
"""

import os
import glob
from pathlib import Path
try:
    from osgeo import gdal, ogr, osr
    gdal.UseExceptions()  # 启用GDAL异常处理
except ImportError:
    print("错误：未找到GDAL库。请安装GDAL：pip install GDAL")
    exit(1)

import geopandas as gpd
from shapely.geometry import box
import pandas as pd
from typing import Union, List, Optional, Tuple
from tqdm import tqdm


def get_tif_bounds_gdal(tif_path: str) -> Optional[Tuple]:
    """
    使用GDAL获取TIF文件的地理边界
    
    Args:
        tif_path (str): TIF文件路径
        
    Returns:
        Optional[Tuple]: (minx, miny, maxx, maxy, crs_wkt) 地理边界和坐标系统，失败返回None
    """
    try:
        # 打开数据集
        dataset = gdal.Open(tif_path, gdal.GA_ReadOnly)
        if dataset is None:
            print(f"警告：无法打开文件 {tif_path}")
            return None
        
        # 获取地理变换参数
        geotransform = dataset.GetGeoTransform()
        if geotransform is None:
            print(f"警告：文件 {tif_path} 没有地理变换信息")
            dataset = None
            return None
        
        # 获取图像尺寸
        width = dataset.RasterXSize
        height = dataset.RasterYSize
        
        # 计算边界坐标
        # geotransform = [top_left_x, pixel_width, rotation, top_left_y, rotation, pixel_height]
        top_left_x = geotransform[0]
        pixel_width = geotransform[1]
        top_left_y = geotransform[3]
        pixel_height = geotransform[5]  # 通常是负值
        
        # 计算四个角的坐标
        minx = top_left_x
        maxx = top_left_x + width * pixel_width
        maxy = top_left_y
        miny = top_left_y + height * pixel_height
        
        # 获取坐标系统
        projection = dataset.GetProjection()
        srs = osr.SpatialReference()
        srs.ImportFromWkt(projection)
        
        # 清理资源
        dataset = None
        
        return minx, miny, maxx, maxy, projection
        
    except Exception as e:
        print(f"警告：处理文件 {tif_path} 时出错: {e}")
        return None


def create_bounds_rectangle_gdal(bounds: Tuple, filename: str) -> Optional[dict]:
    """
    根据边界创建矩形几何对象
    
    Args:
        bounds (Tuple): (minx, miny, maxx, maxy, crs_wkt) 边界坐标
        filename (str): 文件名
        
    Returns:
        Optional[dict]: 包含几何对象和属性的字典，失败返回None
    """
    if bounds is None:
        return None
    
    minx, miny, maxx, maxy, crs_wkt = bounds
    rectangle = box(minx, miny, maxx, maxy)
    
    return {
        'geometry': rectangle,
        'filename': filename,
        'minx': minx,
        'miny': miny,
        'maxx': maxx,
        'maxy': maxy,
        'width': maxx - minx,
        'height': maxy - miny,
        'area': (maxx - minx) * (maxy - miny),
        'crs_wkt': crs_wkt
    }


def get_epsg_from_wkt(wkt: str) -> Optional[str]:
    """
    从WKT字符串中提取EPSG代码
    
    Args:
        wkt (str): WKT格式的坐标系统字符串
        
    Returns:
        Optional[str]: EPSG代码，如"EPSG:4326"，失败返回None
    """
    try:
        srs = osr.SpatialReference()
        srs.ImportFromWkt(wkt)
        authority = srs.GetAuthorityName(None)
        code = srs.GetAuthorityCode(None)
        if authority and code:
            return f"{authority}:{code}"
        return None
    except:
        return None


def generate_tif_bounds_shapefile_gdal(
    tif_folder: Union[str, Path],
    output_shp: Union[str, Path],
    recursive: bool = True,
    target_crs: str = None
) -> Optional[gpd.GeoDataFrame]:
    """
    使用GDAL为文件夹中的所有.tif文件生成边界shapefile
    
    Args:
        tif_folder (Union[str, Path]): 包含.tif文件的文件夹路径
        output_shp (Union[str, Path]): 输出shapefile的路径
        recursive (bool): 是否递归搜索子文件夹，默认True
        target_crs (str): 目标坐标系统，如果None则使用第一个文件的CRS
        
    Returns:
        Optional[gpd.GeoDataFrame]: 生成的GeoDataFrame，如果失败返回None
    """
    tif_folder = Path(tif_folder)
    output_shp = Path(output_shp)
    
    # 确保输出目录存在
    output_shp.parent.mkdir(parents=True, exist_ok=True)
    
    # 查找所有.tif文件
    if recursive:
        tif_pattern = str(tif_folder / "**" / "*.tif")
        tif_files = glob.glob(tif_pattern, recursive=True)
        tif_pattern2 = str(tif_folder / "**" / "*.tiff")
        tif_files.extend(glob.glob(tif_pattern2, recursive=True))
    else:
        tif_pattern = str(tif_folder / "*.tif")
        tif_files = glob.glob(tif_pattern)
        tif_pattern2 = str(tif_folder / "*.tiff")
        tif_files.extend(glob.glob(tif_pattern2))
    
    if not tif_files:
        print(f"在文件夹 {tif_folder} 中未找到.tif文件")
        return None
    
    print(f"找到 {len(tif_files)} 个TIF文件")
    
    # 收集所有边界信息
    rectangles = []
    first_crs_wkt = None
    
    for tif_file in tqdm(tif_files, desc="处理TIF文件"):
        filename = os.path.basename(tif_file)
        bounds_info = get_tif_bounds_gdal(tif_file)
        
        if bounds_info is None:
            continue
            
        # 记录第一个文件的CRS作为默认CRS
        if first_crs_wkt is None:
            first_crs_wkt = bounds_info[4]
        
        rectangle_data = create_bounds_rectangle_gdal(bounds_info, filename)
        if rectangle_data is not None:
            rectangle_data['file_path'] = tif_file
            # 尝试获取EPSG代码
            epsg_code = get_epsg_from_wkt(bounds_info[4])
            rectangle_data['epsg'] = epsg_code if epsg_code else "Unknown"
            rectangles.append(rectangle_data)
    
    if not rectangles:
        print("没有成功处理任何TIF文件")
        return None
    
    # 创建GeoDataFrame
    gdf = gpd.GeoDataFrame(rectangles)
    
    # 设置坐标系统
    if target_crs is not None:
        gdf.crs = target_crs
    elif first_crs_wkt is not None:
        # 尝试从第一个文件的WKT设置CRS
        try:
            epsg_code = get_epsg_from_wkt(first_crs_wkt)
            if epsg_code:
                gdf.crs = epsg_code
            else:
                # 直接使用WKT
                gdf.crs = first_crs_wkt
        except:
            print("警告：无法设置坐标系统")
    else:
        print("警告：无法确定坐标系统")
    
    # 如果指定了目标CRS且与当前CRS不同，进行转换
    if target_crs and str(gdf.crs) != target_crs:
        try:
            print(f"将坐标系统从 {gdf.crs} 转换到 {target_crs}")
            gdf = gdf.to_crs(target_crs)
        except Exception as e:
            print(f"坐标系统转换失败: {e}")
    
    # 保存shapefile
    try:
        gdf.to_file(output_shp, driver='ESRI Shapefile')
        print(f"成功生成shapefile: {output_shp}")
        print(f"包含 {len(gdf)} 个TIF文件的边界")
        
        # 输出统计信息
        print("\n统计信息:")
        print(f"总面积: {gdf['area'].sum():.2f} 平方单位")
        print(f"平均面积: {gdf['area'].mean():.2f} 平方单位")
        print(f"最小面积: {gdf['area'].min():.2f} 平方单位")
        print(f"最大面积: {gdf['area'].max():.2f} 平方单位")
        
        # 输出边界范围
        total_bounds = gdf.total_bounds
        print(f"总边界: minx={total_bounds[0]:.6f}, miny={total_bounds[1]:.6f}, "
              f"maxx={total_bounds[2]:.6f}, maxy={total_bounds[3]:.6f}")
        
        # 输出坐标系统信息
        unique_epsg = gdf['epsg'].unique()
        print(f"包含的坐标系统: {list(unique_epsg)}")
        
        return gdf
        
    except Exception as e:
        print(f"保存shapefile时出错: {e}")
        return None


def main(tif_folder=None, output_shp=None, recursive=True, target_crs=None):
    """
    主函数 - 示例用法
    
    Args:
        tif_folder (str): 包含TIF文件的文件夹路径
        output_shp (str): 输出shapefile路径
        recursive (bool): 是否递归搜索子文件夹，默认True
        target_crs (str): 目标坐标系统，如"EPSG:4326"
    """
    # 如果没有提供参数，使用默认示例值
    if tif_folder is None:
        tif_folder = "./test_tif_data"  # 修改为您的TIF文件夹路径
        print(f"使用默认TIF文件夹: {tif_folder}")
    
    if output_shp is None:
        output_shp = "./output/tif_bounds.shp"  # 修改为您想要的输出路径
        print(f"使用默认输出路径: {output_shp}")
    
    print(f"TIF文件夹: {tif_folder}")
    print(f"输出shapefile: {output_shp}")
    print(f"递归搜索: {recursive}")
    print(f"目标坐标系: {target_crs}")
    print("-" * 50)
    
    # 执行主要功能
    result = generate_tif_bounds_shapefile_gdal(
        tif_folder=tif_folder,
        output_shp=output_shp,
        recursive=recursive,
        target_crs=target_crs
    )
    
    if result is not None:
        print("处理完成！")
        return result
    else:
        print("处理失败！")
        return None


if __name__ == "__main__":
    # 可以选择运行以下任一函数：
    
    # 运行默认示例
    main(tif_folder="/mnt/data/RSIPAC_25_T1/For_Contestants/val/2_Opt", output_shp="/mnt/data/RSIPAC_25_T1/For_Contestants/val_tif_bounds.shp", recursive=True, target_crs="EPSG:4326")