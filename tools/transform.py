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


def reproject_shapefile(shapefile_path, target_tif_path):
    """
    将shapefile文件的参考系转换为目标tif文件的参考系，并返回转换后的GeoDataFrame

    参数:
    shapefile_path (str): 输入shapefile文件路径
    target_tif_path (str): 目标tif文件路径，用于获取参考系

    返回:
    gpd.GeoDataFrame: 转换后的GeoDataFrame
    """
    # 读取shapefile文件
    gdf = gpd.read_file(shapefile_path)
    
    # 获取源shapefile的参考系
    src_crs = gdf.crs
    
    # 打开目标tif文件并获取其投影信息
    target_ds = gdal.Open(target_tif_path)
    target_wkt = target_ds.GetProjection()
    
    # 创建目标参考系
    target_srs = osr.SpatialReference()
    target_srs.ImportFromWkt(target_wkt)
    
    # 将目标参考系转换为GeoPandas的CRS格式
    target_crs = target_srs.ExportToProj4()
    
    # 将GeoDataFrame转换到目标参考系
    gdf = gdf.to_crs(target_crs)
    
    return gdf


# 获取影像的某些波段，保存为新的影像
def get_bands(tif_path, band_list, save_path):
    """
    从多波段影像中提取指定波段并保存为新的影像文件
    
    参数:
        tif_path (str): 输入影像文件路径
        band_list (list): 要提取的波段列表，例如[1,3,4]表示提取第1,3,4波段
        save_path (str): 输出影像文件保存路径
    
    返回:
        str: 保存的影像文件路径
    """
    # 打开输入影像
    dataset = gdal.Open(tif_path)
    if dataset is None:
        raise ValueError(f"无法打开影像文件: {tif_path}")
    
    # 获取影像信息
    proj = dataset.GetProjection()
    geotrans = dataset.GetGeoTransform()
    rows = dataset.RasterYSize
    cols = dataset.RasterXSize
    
    # 获取第一个波段的数据类型，用于保持数据类型一致性
    first_band = dataset.GetRasterBand(band_list[0])
    data_type = first_band.DataType
    
    # 创建输出影像，使用与原始数据相同的数据类型
    driver = gdal.GetDriverByName('GTiff')
    out_dataset = driver.Create(save_path, cols, rows, len(band_list), data_type)
    out_dataset.SetProjection(proj)
    out_dataset.SetGeoTransform(geotrans)
    
    # 读取并写入指定波段
    for i, band_index in enumerate(band_list):
        if band_index <= 0 or band_index > dataset.RasterCount:
            raise ValueError(f"波段索引 {band_index} 超出范围 (1-{dataset.RasterCount})")
        
        # 读取波段数据
        band_data = dataset.GetRasterBand(band_index).ReadAsArray()
        
        # 写入到输出影像
        out_band = out_dataset.GetRasterBand(i+1)
        out_band.WriteArray(band_data)
        
        # 复制统计信息和颜色表（如果有）
        src_band = dataset.GetRasterBand(band_index)
        if src_band.GetStatistics(0, 0) != (0.0, 0.0, 0.0, 0.0):
            stats = src_band.GetStatistics(0, 1)
            out_band.SetStatistics(stats[0], stats[1], stats[2], stats[3])
        
        color_table = src_band.GetColorTable()
        if color_table:
            out_band.SetColorTable(color_table)
    
    # 关闭数据集
    out_dataset.FlushCache()
    del out_dataset
    del dataset
    
    print(f"已成功提取波段 {band_list} 并保存到 {save_path}")
    return save_path


# 批量获取影像的某些波段，保存为新的影像
def get_bands_batch(tif_folder, band_list, save_folder):
    """
    批量处理文件夹中的所有TIF影像，提取指定波段并保存为新的影像
    
    参数:
        tif_folder: 包含TIF影像的文件夹路径
        band_list: 要提取的波段列表，例如 [1, 2, 3]
        save_folder: 保存结果的文件夹路径
    
    返回:
        处理的文件数量
    """
    # 确保保存文件夹存在
    os.makedirs(save_folder, exist_ok=True)  # exist_ok=True 表示如果文件夹存在，不报错
    
    # 获取所有tif文件
    tif_files = [f for f in os.listdir(tif_folder) if f.lower().endswith(('.tif', '.tiff', '.TIF', '.TIFF'))]
    
    if not tif_files:
        print(f"在 {tif_folder} 中未找到TIF文件")
        return 0
    
    processed_count = 0
    for tif_file in tif_files:
        input_path = os.path.join(tif_folder, tif_file)
        output_path = os.path.join(save_folder, tif_file)
        
        try:
            get_bands(input_path, band_list, output_path)
            processed_count += 1
            print(f"已处理 {processed_count}/{len(tif_files)}: {tif_file}")
        except Exception as e:
            print(f"处理 {tif_file} 时出错: {str(e)}")
    
    print(f"批量处理完成，共处理了 {processed_count} 个文件")



if __name__ == '__main__':
    target_tif_path = '/mnt/data/TreeHeight/nDSM_cropped_boundary.tif'
    shapefile_path = '/mnt/data/TreeHeight/pic4Draw.gdb'
    reprojected_gdf = reproject_shapefile(shapefile_path, target_tif_path)
    # 保存为GDB文件
    # 由于FileGDB驱动不支持，需要先安装GDAL和fiona库的FileGDB支持
    # 可以通过以下命令安装:
    # conda install -c conda-forge gdal fiona
    # 或者使用其他格式如GeoPackage作为替代
    reprojected_gdf.to_file('/mnt/data/TreeHeight/pic4Draw_reprojected.gpkg', driver='GPKG')


