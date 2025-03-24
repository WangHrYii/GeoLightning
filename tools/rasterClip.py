import os
from osgeo import gdal, ogr, osr
from transform import reproject_tif, reproject_shapefile
from tqdm import tqdm
import numpy as np

def clip_raster_with_shapefile_v1(raster_path, shapefile_path, output_path):
    """
    根据矢量文件裁剪栅格文件,采用shapefile的参考系
    """
    # 1.栅格文件重新投影，保证栅格文件和矢量文件在同一坐标系下
    # 读取矢量文件
    shapefile = ogr.Open(shapefile_path)
    # 获取矢量文件的投影信息
    shapefile_proj = shapefile.GetLayer().GetSpatialRef().ExportToWkt()  # 得到的是wkt格式的字符串
    # 读取栅格文件
    raster = gdal.Open(raster_path)
    # 获取栅格文件的投影信息
    raster_proj = raster.GetProjection()  # 得到的是wkt格式的字符串

    sr1 = osr.SpatialReference()
    sr1.ImportFromWkt(shapefile_proj)

    sr2 = osr.SpatialReference()
    sr2.ImportFromWkt(raster_proj)

    # 判断两个投影是否相同
    if sr1.IsSame(sr2):
        reproject_tif_path = raster_path
    else:
        reproject_tif_path = reproject_tif(raster_path, shapefile_proj)
    # 2.重新读取栅格文件
    raster = gdal.Open(reproject_tif_path)
    # 3.读取矢量文件
    shapefile = ogr.Open(shapefile_path)
    shapefile_layer = shapefile.GetLayer()

    # 4.栅格裁剪
    gdal.Warp(output_path,
              raster,
              format='GTiff',
              cutlineDSName=shapefile_path,
              cropToCutline = True,
              dstNodata=0)


def clip_raster_with_shapefile_v2(raster_path, shapefile_path, output_path):
    """
    根据矢量文件裁剪栅格文件,采用栅格文件的参考系
    """
    # 1.矢量文件重新投影，保证栅格文件和矢量文件在同一坐标系下
    # 读取矢量文件
    shapefile = ogr.Open(shapefile_path)
    # 获取矢量文件的投影信息
    shapefile_proj = shapefile.GetLayer().GetSpatialRef().ExportToWkt()  # 得到的是wkt格式的字符串
    # 读取栅格文件
    raster = gdal.Open(raster_path)
    # 获取栅格文件的投影信息
    raster_proj = raster.GetProjection()  # 得到的是wkt格式的字符串

    sr1 = osr.SpatialReference()
    sr1.ImportFromWkt(shapefile_proj)

    sr2 = osr.SpatialReference()
    sr2.ImportFromWkt(raster_proj)

    # 判断两个投影是否相同
    reproject_shapefile_gdf = None
    if sr1.IsSame(sr2):
        reproject_shapefile_path = shapefile_path
    else:
        # 重新投影矢量文件，使得矢量文件和栅格文件在同一坐标系下
        reproject_shapefile_gdf = reproject_shapefile(shapefile_path, raster_proj)
        reproject_shapefile_path = shapefile_path.replace('.shp', '_reprojected.shp')
        reproject_shapefile_gdf.to_file(reproject_shapefile_path)
    
    # 2.裁剪
    gdal.Warp(output_path,
              raster,
              format='GTiff',
              cutlineDSName=reproject_shapefile_path,
              cropToCutline = True,
              dstNodata=0)


# 读取tif数据集
def readTif(fileName):
    """tif文件读取函数
    fileName: 文件路径
    return: gdal.Dataset
    """
    dataset = gdal.Open(fileName)
    assert dataset!=None, f'File not found: {fileName}'
    return dataset
    
# 保存tif文件函数
def writeTiff(im_data, im_geotrans, im_proj, path):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    elif len(im_data.shape) == 2:
        im_data = np.array([im_data])
        im_bands, im_height, im_width = im_data.shape
    # 创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, int(im_width), int(im_height), int(im_bands), datatype)
    if(dataset!= None):
        dataset.SetGeoTransform(im_geotrans) # 写入仿射变换参数
        dataset.SetProjection(im_proj) # 写入投影
    for i in range(im_bands):
        dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset
    
# 像素坐标和地理坐标仿射变换
def CoordTransf(Xpixel, Ypixel, GeoTransform):
    XGeo = GeoTransform[0]+GeoTransform[1]*Xpixel+Ypixel*GeoTransform[2]
    YGeo = GeoTransform[3]+GeoTransform[4]*Xpixel+Ypixel*GeoTransform[5]
    return XGeo, YGeo


def TifCrop(TifPath, SavePath, CropSize, RepetitionRate):
    """单张Tif影像裁剪函数，支持多波段影像
    TifPath: 输入影像路径
    SavePath: 输出影像保存路径
    CropSize: 裁剪尺寸
    RepetitionRate: 重复率
    TODO: 2.现在必须输入完整的save路径，需要优化成只输入文件夹路径，自动创建文件夹
          3.改写为rasterio版本
    """
    if not os.path.exists(SavePath): os.makedirs(SavePath)
    
    dataset_img = readTif(TifPath)
    width = dataset_img.RasterXSize
    height = dataset_img.RasterYSize
    proj = dataset_img.GetProjection()
    geotrans = dataset_img.GetGeoTransform()

    row_count = int((height - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))
    col_count = int((width - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))

    num = row_count * col_count
    with tqdm(total=num, desc=f"裁剪进度") as pbar:
        # 裁剪图片,重复率为RepetitionRate
        for i in range(row_count):
            for j in range(col_count):
                x_offset = int(j * CropSize * (1 - RepetitionRate))
                y_offset = int(i * CropSize * (1 - RepetitionRate))
                x_size = min(CropSize, width - x_offset)
                y_size = min(CropSize, height - y_offset)

                img = dataset_img.ReadAsArray(x_offset, y_offset, x_size, y_size)
                
                XGeo, YGeo = CoordTransf(x_offset, y_offset, geotrans)
                crop_geotrans = (XGeo, geotrans[1], geotrans[2], YGeo, geotrans[4], geotrans[5])
                # 写图像
                writeTiff(img, crop_geotrans, proj, os.path.join(SavePath, f"{i}_{j}.tif"))
                pbar.update(1)

        # 向前裁剪最后一列
        for i in range(row_count):
            x_offset = width - CropSize
            y_offset = int(i * CropSize * (1 - RepetitionRate))
            x_size = CropSize
            y_size = min(CropSize, height - y_offset)
            
            img = dataset_img.ReadAsArray(x_offset, y_offset, x_size, y_size)
            
            XGeo, YGeo = CoordTransf(x_offset, y_offset, geotrans)
            crop_geotrans = (XGeo, geotrans[1], geotrans[2], YGeo, geotrans[4], geotrans[5])
            # 写图像
            writeTiff(img, crop_geotrans, proj, os.path.join(SavePath, f"{i}_{col_count}.tif"))
            pbar.update(1)

        # 向前裁剪最后一行
        for j in range(col_count):
            x_offset = int(j * CropSize * (1 - RepetitionRate))
            y_offset = height - CropSize
            x_size = min(CropSize, width - x_offset)
            y_size = CropSize
            
            img = dataset_img.ReadAsArray(x_offset, y_offset, x_size, y_size)
            
            XGeo, YGeo = CoordTransf(x_offset, y_offset, geotrans)
            crop_geotrans = (XGeo, geotrans[1], geotrans[2], YGeo, geotrans[4], geotrans[5])
            writeTiff(img, crop_geotrans, proj, os.path.join(SavePath, f"{row_count}_{j}.tif"))
            pbar.update(1)

        # 裁剪右下角
        x_offset = width - CropSize
        y_offset = height - CropSize
        x_size = CropSize
        y_size = CropSize
        
        img = dataset_img.ReadAsArray(x_offset, y_offset, x_size, y_size)
        
        XGeo, YGeo = CoordTransf(x_offset, y_offset, geotrans)
        crop_geotrans = (XGeo, geotrans[1], geotrans[2], YGeo, geotrans[4], geotrans[5])
        writeTiff(img, crop_geotrans, proj, os.path.join(SavePath, f"{row_count}_{col_count}.tif"))
        pbar.update(1)



# # 示例调用
# raster_path = '/mnt/data/Tree/TreeHeight/tree_cover/tree_reprojected.tif'
# shapefile_path = '/mnt/data/Tree/TreeHeight/boundary/boundary.shp'
# output_path = '/mnt/data/Tree/TreeHeight/tree_cover/tree_crop_bound_reproj.tif'

# clip_raster_with_shapefile_v2(raster_path, shapefile_path, output_path)