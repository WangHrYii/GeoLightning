#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from pathlib import Path
from typing import List, Optional, Union

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, Polygon, LineString
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ShapefileMerger:
    """Shapefile合并器类"""
    
    def __init__(self):
        self.supported_formats = ['.shp', '.geojson', '.gpkg']
    
    def validate_file(self, file_path: Union[str, Path]) -> bool:
        """验证文件是否存在且格式正确"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.error(f"文件不存在: {file_path}")
            return False
        
        if file_path.suffix.lower() not in self.supported_formats:
            logger.error(f"不支持的文件格式: {file_path.suffix}")
            return False
        
        return True
    
    def read_shapefile(self, file_path: Union[str, Path]) -> Optional[gpd.GeoDataFrame]:
        """读取shapefile文件"""
        try:
            gdf = gpd.read_file(file_path)
            logger.info(f"成功读取文件: {file_path}, 记录数: {len(gdf)}")
            return gdf
        except Exception as e:
            logger.error(f"读取文件失败 {file_path}: {str(e)}")
            return None
    
    def merge_simple(self, gdfs: List[gpd.GeoDataFrame]) -> gpd.GeoDataFrame:
        """简单合并 - 直接拼接所有数据"""
        try:
            # 确保所有GeoDataFrame使用相同的坐标系
            crs = gdfs[0].crs
            for i, gdf in enumerate(gdfs[1:], 1):
                if gdf.crs != crs:
                    logger.warning(f"第{i+1}个文件的坐标系不同，正在转换为: {crs}")
                    gdfs[i] = gdf.to_crs(crs)
            
            # 合并数据
            merged = pd.concat(gdfs, ignore_index=True)
            merged = gpd.GeoDataFrame(merged, crs=crs)
            
            logger.info(f"合并完成，总记录数: {len(merged)}")
            return merged
            
        except Exception as e:
            logger.error(f"合并失败: {str(e)}")
            raise
    
    def merge_with_intersection(self, gdf1: gpd.GeoDataFrame, gdf2: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """基于空间相交的合并"""
        try:
            # 确保坐标系一致
            if gdf1.crs != gdf2.crs:
                gdf2 = gdf2.to_crs(gdf1.crs)
            
            # 执行空间连接
            intersection = gpd.overlay(gdf1, gdf2, how='intersection')
            
            logger.info(f"空间相交合并完成，结果记录数: {len(intersection)}")
            return intersection
            
        except Exception as e:
            logger.error(f"空间相交合并失败: {str(e)}")
            raise
    
    def merge_with_union(self, gdf1: gpd.GeoDataFrame, gdf2: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """基于空间联合的合并"""
        try:
            # 确保坐标系一致
            if gdf1.crs != gdf2.crs:
                gdf2 = gdf2.to_crs(gdf1.crs)
            
            # 执行空间联合
            union = gpd.overlay(gdf1, gdf2, how='union')
            
            logger.info(f"空间联合合并完成，结果记录数: {len(union)}")
            return union
            
        except Exception as e:
            logger.error(f"空间联合合并失败: {str(e)}")
            raise
    
    def save_shapefile(self, gdf: gpd.GeoDataFrame, output_path: Union[str, Path], 
                      driver: str = 'ESRI Shapefile') -> bool:
        """保存合并后的shapefile"""
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 根据文件扩展名确定驱动器
            if output_path.suffix.lower() == '.geojson':
                driver = 'GeoJSON'
            elif output_path.suffix.lower() == '.gpkg':
                driver = 'GPKG'
            
            gdf.to_file(output_path, driver=driver)
            logger.info(f"文件保存成功: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"保存文件失败: {str(e)}")
            return False
    
    def merge_files(self, input_files: List[Union[str, Path]], output_file: Union[str, Path],
                   merge_type: str = 'simple') -> bool:
        """主要的文件合并函数"""
        logger.info(f"开始合并 {len(input_files)} 个文件")
        
        # 验证输入文件
        for file_path in input_files:
            if not self.validate_file(file_path):
                return False
        
        # 读取所有文件
        gdfs = []
        for file_path in input_files:
            gdf = self.read_shapefile(file_path)
            if gdf is None:
                return False
            gdfs.append(gdf)
        
        # 根据合并类型执行合并
        try:
            if merge_type == 'simple':
                merged_gdf = self.merge_simple(gdfs)
            elif merge_type == 'intersection':
                if len(gdfs) != 2:
                    logger.error("空间相交合并只支持两个文件")
                    return False
                merged_gdf = self.merge_with_intersection(gdfs[0], gdfs[1])
            elif merge_type == 'union':
                if len(gdfs) != 2:
                    logger.error("空间联合合并只支持两个文件")
                    return False
                merged_gdf = self.merge_with_union(gdfs[0], gdfs[1])
            else:
                logger.error(f"不支持的合并类型: {merge_type}")
                return False
            
            # 保存结果
            return self.save_shapefile(merged_gdf, output_file)
            
        except Exception as e:
            logger.error(f"合并过程出错: {str(e)}")
            return False


def merge_shapefiles(input_files: List[str], output_file: str, merge_type: str = 'simple') -> bool:
    """
    便捷函数：合并shapefile文件
    
    Args:
        input_files: 输入文件路径列表
        output_file: 输出文件路径
        merge_type: 合并类型 ('simple', 'intersection', 'union')
    
    Returns:
        bool: 是否合并成功
    
    Example:
        # 简单合并两个shapefile
        merge_shapefiles(['file1.shp', 'file2.shp'], 'merged.shp')
        
        # 空间相交合并
        merge_shapefiles(['file1.shp', 'file2.shp'], 'result.shp', 'intersection')
    """
    merger = ShapefileMerger()
    return merger.merge_files(input_files, output_file, merge_type)


if __name__ == '__main__':
    # 测试合并
    input_files = ['/home/whr/LocalData/PKU_Industry/industry.shp', '/home/whr/LocalData/PKU_Industry/GHM.shp']
    output_file = '/home/whr/LocalData/PKU_Industry/data/Result_v1/industry.shp'
    merge_type = 'simple'
    merge_shapefiles(input_files, output_file, merge_type)
