# Function: Convert shapefile to geojsons for downloading
# Author: Wang Haoran

import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
from rtree import index as rindex
import networkx as nx
from shapely.ops import unary_union
import math
import os
import pandas as pd

def get_utm_zone(geom):
    centroid = geom.centroid
    longitude = centroid.x
    latitude = centroid.y
    utm_zone = math.floor((longitude + 180) / 6) + 1
    if latitude >= 0:
        epsg_code = f'EPSG:326{utm_zone}'
    else:
        epsg_code = f'EPSG:327{utm_zone}'
    return epsg_code

def convert_shapefile_to_geojsons(shapefile_path, output_folder):
    # 读取Shapefile
    gdf = gpd.read_file(shapefile_path)
    
    # 确保只处理Polygon和MultiPolygon
    gdf = gdf[gdf.geometry.type.isin(['Polygon', 'MultiPolygon'])]
    
    # # 自动选择UTM投影
    # # 取数据的中心点所在的UTM区
    # center_geom = gdf.unary_union.centroid
    # utm_crs = get_utm_zone(center_geom)
    
    # # 投影到UTM坐标系
    # gdf_utm = gdf.to_crs(utm_crs)

    gdf_utm = gdf
    
    # 进行5公里缓冲区操作
    buffer_distance = 2000  # 米
    gdf_buffered = gdf_utm.buffer(buffer_distance)
    
    # 投影回WGS84
    gdf_buffered_wgs84 = gdf_buffered.to_crs('EPSG:4326')

    # 将MultiPolygon分解为单个的Polygon
    gdf = gdf_buffered_wgs84.explode(index_parts=False)

    # 创建R树索引
    idx = rindex.Index()
    for i, geom in enumerate(gdf.geometry):
        idx.insert(i, geom.bounds)

    # 创建图
    G = nx.Graph()
    G.add_nodes_from(range(len(gdf)))

    # 检查重叠并添加边
    for i in range(len(gdf)):
        possible_overlaps = list(idx.intersection(gdf.geometry[i].bounds))
        for j in possible_overlaps:
            if i != j and gdf.geometry[i].intersects(gdf.geometry[j]):
                G.add_edge(i, j)

    # 找到所有的连通分量
    connected_components = list(nx.connected_components(G))

    # 合并每个连通分量内的Polygon
    merged_polys = []
    for component in connected_components:
        component_polys = gdf.loc[list(component)].geometry
        merged_poly = unary_union(component_polys)
        merged_polys.append(merged_poly)

    # 创建新的GeoDataFrame
    new_gdf = gpd.GeoDataFrame(geometry=merged_polys, crs=gdf.crs)
    
    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 整个gdf_buffered_wgs84保存为GeoJSON文件
    output_file = os.path.join(output_folder, 'buffered.geojson')
    new_gdf.to_file(output_file, driver='GeoJSON')
    
    # # 提取每个缓冲区的bbox并输出为GeoJSON文件
    # for index, geom in enumerate(new_gdf.geometry):
    #     if geom is not None and isinstance(geom, (Polygon, MultiPolygon)):
    #         minx, miny, maxx, maxy = geom.bounds
    #         # 创建bbox Polygon
    #         bbox_poly = Polygon([(minx, miny), (minx, maxy), (maxx, maxy), (maxx, miny), (minx, miny)])
    #         # 创建新的GeoDataFrame
    #         gdf_bbox = gpd.GeoDataFrame({'id': [index]}, geometry=[bbox_poly], crs='EPSG:4326')
    #         # 输出为GeoJSON文件
    #         output_file = os.path.join(output_folder, f'bbox_{index}.geojson')
    #         gdf_bbox.to_file(output_file, driver='GeoJSON')
    #     else:
    #         print(f"Geometry at index {index} is invalid or not a Polygon/MultiPolygon.")

    # 将所有的bbox合并为一个shp文件
    gdf_bbox_all = []

    for index, geom in enumerate(new_gdf.geometry):
        if geom is not None and isinstance(geom, (Polygon, MultiPolygon)):
            minx, miny, maxx, maxy = geom.bounds
            # 创建bbox Polygon
            bbox_poly = Polygon([(minx, miny), (minx, maxy), (maxx, maxy), (maxx, miny), (minx, miny)])
            # 创建新的GeoDataFrame
            gdf_bbox = gpd.GeoDataFrame({'id': [index]}, geometry=[bbox_poly], crs='EPSG:4326')
            gdf_bbox_all.append(gdf_bbox)
        else:
            print(f"Geometry at index {index} is invalid or not a Polygon/MultiPolygon.")

    gdf_bbox_all = gpd.GeoDataFrame(pd.concat(gdf_bbox_all, ignore_index=True))
    output_folder_shp = os.path.join(output_folder.replace('geojsons', 'shp'))
    if not os.path.exists(output_folder_shp):
        os.makedirs(output_folder_shp)
    # 输出为Shapefile文件
    output_file = os.path.join(output_folder_shp, 'bbox_all_infra.shp')
    gdf_bbox_all.to_file(output_file)
    

def gen_bbox_point(shapefile_path, output_folder, buffer_distance):
    """
    将输入Shapefile中的点生成缓冲区，融合重叠区域后创建边界框，输出为WGS84坐标系
    
    参数:
    shapefile_path (str): 输入点要素文件路径
    output_folder (str): 输出文件夹路径
    buffer_distance (int): 缓冲距离（米）
    """
    # 读取Shapefile
    gdf = gpd.read_file(shapefile_path)
    
    # 确保只处理Polygon和MultiPolygon
    gdf = gdf[gdf.geometry.type.isin(['Point'])]

    # TODO: 选择合适的UTM投影, 这里直接选择EPSG:32650
    utm_epsg = 'EPSG:32650'

    # 转换到UTM坐标系
    gdf_utm = gdf.to_crs(utm_epsg)
    
    # 进行缓冲区操作
    gdf_buffered = gdf_utm.buffer(buffer_distance)
    
    # 投影回WGS84
    gdf_buffered_wgs84 = gdf_buffered.to_crs('EPSG:4326')

    # 将MultiPolygon分解为单个的Polygon
    gdf = gdf_buffered_wgs84.explode(index_parts=False)

    # 创建R树索引
    idx = rindex.Index()
    for i, geom in enumerate(gdf.geometry):
        idx.insert(i, geom.bounds)

    # 创建图
    G = nx.Graph()
    G.add_nodes_from(range(len(gdf)))

    # 检查重叠并添加边
    for i in range(len(gdf)):
        possible_overlaps = list(idx.intersection(gdf.geometry[i].bounds))
        for j in possible_overlaps:
            if i != j and gdf.geometry[i].intersects(gdf.geometry[j]):
                G.add_edge(i, j)

    # 找到所有的连通分量
    connected_components = list(nx.connected_components(G))

    # 合并每个连通分量内的Polygon
    merged_polys = []
    for component in connected_components:
        component_polys = gdf.loc[list(component)].geometry
        merged_poly = unary_union(component_polys)
        merged_polys.append(merged_poly)

    # 创建新的GeoDataFrame
    new_gdf = gpd.GeoDataFrame(geometry=merged_polys, crs=gdf.crs)
    
    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 将所有的bbox合并为一个shp文件
    gdf_bbox_all = []

    for index, geom in enumerate(new_gdf.geometry):
        if geom is not None and isinstance(geom, (Polygon, MultiPolygon)):
            minx, miny, maxx, maxy = geom.bounds
            # 创建bbox Polygon
            bbox_poly = Polygon([(minx, miny), (minx, maxy), (maxx, maxy), (maxx, miny), (minx, miny)])
            # 创建新的GeoDataFrame
            gdf_bbox = gpd.GeoDataFrame({'id': [index]}, geometry=[bbox_poly], crs='EPSG:4326')
            gdf_bbox_all.append(gdf_bbox)
        else:
            print(f"Geometry at index {index} is invalid or not a Polygon/MultiPolygon.")

    gdf_bbox_all = gpd.GeoDataFrame(pd.concat(gdf_bbox_all, ignore_index=True))
    # 输出为Shapefile文件
    output_file = os.path.join(output_folder, 'bbox_all_infra.shp')
    gdf_bbox_all.to_file(output_file)
    

if __name__ == '__main__':
    # shapefile_path = '/mnt/data_1/Industry/Truth/44_14_51_3.shp'
    # output_folder = '/mnt/data_1/Industry/Truth/geojsons_2'
    # convert_shapefile_to_geojsons(shapefile_path, output_folder)
    shapefile_path = '/mnt/data_1/Industry/Truth/Industry_center/industry_center.shp'
    output_folder = '/mnt/data_1/Industry/Truth/industry_bbox'
    buffer_distance = 5000
    gen_bbox_point(shapefile_path, output_folder, buffer_distance)