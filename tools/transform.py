import os
import geopandas as gpd
from osgeo import ogr, gdal, osr


def get_tif_meta(tif_path):
    dataset = gdal.Open(tif_path)
    # 栅格矩阵的列数
    width = dataset.RasterXSize 
    # 栅格矩阵的行数
    height = dataset.RasterYSize 
    # 获取仿射矩阵信息
    geotrans = dataset.GetGeoTransform()
    # 获取投影信息
    proj = dataset.GetProjection()
    return width, height, geotrans, proj

def shp2tif(shp_path, refer_tif_path, target_tif_path, data_value=1, nodata_value=0):

    width, height, geotrans, proj = get_tif_meta(refer_tif_path)
    # 读取shp文件
    shp_file = ogr.Open(shp_path)
    # 获取图层文件对象
    shp_layer = shp_file.GetLayer()
    # 创建栅格
    target_ds = gdal.GetDriverByName('GTiff').Create(
        utf8_path = target_tif_path,    # 栅格地址
        xsize = width,                  # 栅格宽
        ysize = height,                 # 栅格高
        bands = 1,                      # 栅格波段数
        eType = gdal.GDT_Byte           # 栅格数据类型
        )
    # 将参考栅格的仿射变换信息设置为结果栅格仿射变换信息
    target_ds.SetGeoTransform(geotrans)
    # 设置投影坐标信息
    target_ds.SetProjection(proj)
    band = target_ds.GetRasterBand(1)
    # 设置背景nodata数值
    band.SetNoDataValue(nodata_value)
    band.FlushCache()
    
    # 栅格化函数
    gdal.RasterizeLayer(
        dataset = target_ds,                        # 输出的栅格数据集
        bands = [1],                                # 输出波段
        layer = shp_layer,                          # 输入待转换的矢量图层
        burn_values = [data_value],                 # 烧录值
        )
    
    del target_ds


# 重投影函数
def reproject_tif(input_tif_path, target_epsg):
    # 读取tif文件
    dataset = gdal.Open(input_tif_path)
    # 获取tif文件的投影信息
    proj = dataset.GetProjection()
    # 获取tif文件的仿射变换信息
    geotrans = dataset
    # 创建输出文件
    output_tif_path = input_tif_path.replace('.tif', '_reprojected.tif')
    gdal.Warp(output_tif_path, dataset, dstSRS=f'{target_epsg}')
    return output_tif_path


def reproject_shapefile(shapefile_path, target_wkt):
    """
    将shapefile文件的参考系转换为目标参考系，并返回转换后的GeoDataFrame

    参数:
    shapefile_path (str): 输入shapefile文件路径
    target_wkt (str): 目标参考系的WKT字符串

    返回:
    gpd.GeoDataFrame: 转换后的GeoDataFrame
    """
    # 读取shapefile文件
    gdf = gpd.read_file(shapefile_path)
    
    # 获取源shapefile的参考系
    src_crs = gdf.crs
    
    # 创建目标参考系
    target_srs = osr.SpatialReference()
    target_srs.ImportFromWkt(target_wkt)
    
    # 将目标参考系转换为GeoPandas的CRS格式
    target_crs = target_srs.ExportToProj4()
    
    # 将GeoDataFrame转换到目标参考系
    gdf = gdf.to_crs(target_crs)
    
    return gdf





if __name__ == '__main__':
    raster_path = '/mnt/data/Tree/TreeHeight/tree_cover/tree.tif'
    taget_tif_path = '/mnt/data/Tree/TreeHeight/nDSM.tif'
    save_path = '/mnt/data/Tree/TreeHeight/tree_cover/tree_reproj.tif'

    target_tif_gdal = gdal.Open(taget_tif_path)
    target_tif_proj = target_tif_gdal.GetProjection()

    reproject_tif(raster_path, target_tif_proj)


