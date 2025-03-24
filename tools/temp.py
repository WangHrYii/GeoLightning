import geopandas as gpd
from shapely.geometry import Polygon
import utm
import os

def gen_bbox_point(shapefile_path, output_folder, buffer_distance):
    """
    将输入Shapefile中的点生成缓冲区，融合重叠区域后创建边界框，输出为WGS84坐标系
    
    参数:
    shapefile_path (str): 输入点要素文件路径
    output_folder (str): 输出文件夹路径
    buffer_distance (int): 缓冲距离（米）
    """
    
    # 读取输入数据
    gdf = gpd.read_file(shapefile_path)
    
    # 校验几何类型
    if not all(gdf.geometry.type == 'Point'):
        raise ValueError("输入文件必须为点要素类型")
    
    # 强制转换为WGS84坐标系
    gdf_wgs84 = gdf.to_crs('EPSG:4326') if gdf.crs != 'EPSG:4326' else gdf
    
    # 计算UTM投影参数
    avg_lon = gdf_wgs84.geometry.x.mean()
    avg_lat = gdf_wgs84.geometry.y.mean()
    utm_zone = utm.from_latlon(avg_lat, avg_lon)
    utm_epsg = f'EPSG:{"326" if utm_zone[3] >= "N" else "327"}{utm_zone[2]}'
    
    # 转换到UTM坐标系进行缓冲
    gdf_utm = gdf_wgs84.to_crs(utm_epsg)
    gdf_utm['buffered'] = gdf_utm.geometry.buffer(buffer_distance)
    
    # 融合重叠的缓冲区
    merged = gdf_utm['buffered'].unary_union
    
    # 分解合并后的几何体
    if merged.is_empty:
        merged_polygons = []
    elif merged.geom_type == 'Polygon':
        merged_polygons = [merged]
    else:
        merged_polygons = list(merged.geoms)
    
    # 生成边界框
    bbox_polygons = []
    for poly in merged_polygons:
        minx, miny, maxx, maxy = poly.bounds
        bbox = Polygon([(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)])
        bbox_polygons.append(bbox)
    
    # 创建GeoDataFrame并转换坐标系
    bbox_gdf = gpd.GeoDataFrame(geometry=bbox_polygons, crs=utm_epsg)
    bbox_gdf_wgs84 = bbox_gdf.to_crs('EPSG:4326')
    
    # 确保输出目录存在
    os.makedirs(output_folder, exist_ok=True)
    
    # 保存结果
    output_path = os.path.join(output_folder, "merged_bboxes.shp")
    bbox_gdf_wgs84.to_file(output_path)
    print(f"处理完成，结果已保存至：{output_path}")