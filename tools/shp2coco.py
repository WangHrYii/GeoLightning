import os
from osgeo import gdal
import numpy as np
import geopandas as gpd
import json
from shapely.geometry import mapping, box
from shapely.ops import transform
import fiona
import datetime

from tqdm import tqdm


# 读取tif数据集
def readTif(fileName):
    dataset = gdal.Open(fileName)
    if dataset == None:
        print(fileName + "文件无法打开")
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

def geo_to_pixel_coords(x, y, geotrans):
    """
    将地理坐标转换为像素坐标
    """
    pixel_x = int((x - geotrans[0]) / geotrans[1])
    pixel_y = int((y - geotrans[3]) / geotrans[5])  # geotrans[5] 通常为负数
    return pixel_x, pixel_y

def get_pixel_segmentation(intersection, geotrans, x_offset, y_offset):
    """
    将分割的地理坐标转换为像素坐标
    """
    coords = list(intersection.exterior.coords)
    coords = [(x, y) for x, y, _ in coords]
    pixel_coords = [geo_to_pixel_coords(x, y, geotrans) for x, y in coords]
    pixel_coords = [(px - x_offset, py - y_offset) for px, py in pixel_coords]
    # Flatten the list of tuples
    pixel_coords_flat = [coord for point in pixel_coords for coord in point]
    return pixel_coords_flat


def TifwithShape2COCO(TifPath, ShapefilePath, SavePath, CropSize, RepetitionRate):
    '''
    tif影像及其shapefile转instans coco实例分割, 滑动窗口裁剪
    TifPath 影像路径
    ShapefilePath shapefile路径
    SavePath 裁剪后保存目录
    CropSize 裁剪尺寸
    RepetitionRate 重复率
    '''
    
    if not os.path.exists(SavePath): os.makedirs(SavePath)
    
    dataset_img = gdal.Open(TifPath)
    width = dataset_img.RasterXSize
    height = dataset_img.RasterYSize
    proj = dataset_img.GetProjection()
    geotrans = dataset_img.GetGeoTransform()

    # 读取shapefile
    gdf = gpd.read_file(ShapefilePath)

    # 创建coco实例
    coco = {
        "info": {
            "description": "Tree instance segmentation",
            "url": '',
            "version": "1.0",
            "year": 2025,
            "contributor": "Wang Haoran",
            "date_created": datetime.datetime.utcnow().isoformat(' ')
        },
        "licenses": [
            {
                "url": "",
                "id": 1,
                "name": ""
            }
        ],
        "categories": [
            {
                "supercategory": "tree",
                "id": 1,
                "name": "tree"
            }
        ],
        "images": [],
        "annotations": []
    }
    
    # 获取当前文件夹的文件个数len,并以len+1命名即将裁剪得到的图像
    new_name = len(os.listdir(SavePath)) + 1  # 文件名从1开始
    annotation_id = 1                         # annotation_id从1开始

    num_images = int((height - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate))) * int((width - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate))) + 2

    with tqdm(total=num_images) as pbar:

        # 裁剪图片,重复率为RepetitionRate
        for i in range(int((height - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))):
            for j in range(int((width - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))):
                x_offset = int(j * CropSize * (1 - RepetitionRate))
                y_offset = int(i * CropSize * (1 - RepetitionRate))
                
                cropped = dataset_img.ReadAsArray(x_offset, y_offset, CropSize, CropSize)
                
                XGeo, YGeo = CoordTransf(x_offset, y_offset, geotrans)  # 获取裁剪图像左上角的地理坐标
                crop_geotrans = (XGeo, geotrans[1], geotrans[2], YGeo, geotrans[4], geotrans[5])  # 裁剪图像的仿射变换参数

                img_path = os.path.join(SavePath, f"{new_name}.tif")

                img_info = {
                    "file_name": img_path,
                    "height": CropSize,
                    "width": CropSize,
                    "id": new_name
                }
                # coco实例中添加图像信息
                coco['images'].append(img_info)

                # 获取当前裁剪图像的几何信息
                img_geom = box(XGeo, YGeo, XGeo + CropSize * geotrans[1], YGeo + CropSize * geotrans[5])  # 地理坐标系下的几何信息

                # 遍历shapefile,获取与当前裁剪图像相交的几何信息
                for idx, row in gdf.iterrows():
                    geom = row.geometry
                    if geom.intersects(img_geom):
                        intersection = geom.intersection(img_geom)
                        if not intersection.is_empty:
                            # segmentation = [list(np.array(intersection.exterior.coords).flatten())]  # 获得的是多边形的坐标，坐标系不是像素坐标，需要转换
                            segmentation = [get_pixel_segmentation(intersection, geotrans, x_offset, y_offset)]
                            bbox = list(intersection.bounds)
                            area = intersection.area
                            annotation_info = {
                                "id": annotation_id,
                                "image_id": new_name,
                                "category_id": 1,
                                "segmentation": segmentation,
                                "area": area,
                                "bbox": bbox,
                                "iscrowd": 0
                            }
                            coco["annotations"].append(annotation_info)
                            annotation_id += 1

                # 写图像
                writeTiff(cropped, crop_geotrans, proj, os.path.join(SavePath, f"{new_name}.tif"))
                pbar.update(1)
                # 文件名 + 1
                new_name += 1
        
        # 向前裁剪最后一列
        for i in range(int((height - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))):
            y_offset = int(i * CropSize * (1 - RepetitionRate))
            x_offset = width - CropSize
            cropped = dataset_img.ReadAsArray(x_offset, y_offset, CropSize, CropSize)
            
            XGeo, YGeo = CoordTransf(x_offset, y_offset, geotrans)
            crop_geotrans = (XGeo, geotrans[1], geotrans[2], YGeo, geotrans[4], geotrans[5])

            img_path = os.path.join(SavePath, f"{new_name}.tif")

            img_info = {
                "file_name": img_path,
                "height": CropSize,
                "width": CropSize,
                "id": new_name
            }
            coco['images'].append(img_info)

            # 获取当前裁剪图像的几何信息
            img_geom = box(XGeo, YGeo, XGeo + CropSize * geotrans[1], YGeo + CropSize * geotrans[5])
            img_geom = transform(lambda x, y: (x, y), img_geom)

            # 遍历shapefile,获取与当前裁剪图像相交的几何信息
            for idx, row in gdf.iterrows():
                geom = row.geometry
                if geom.intersects(img_geom):
                    intersection = geom.intersection(img_geom)
                    if not intersection.is_empty:
                        segmentation = [get_pixel_segmentation(intersection, geotrans, x_offset, y_offset)]
                        bbox = list(intersection.bounds)
                        area = intersection.area

                        annotation_info = {
                            "id": annotation_id,
                            "image_id": new_name,
                            "category_id": 1,
                            "segmentation": segmentation,
                            "area": area,
                            "bbox": bbox,
                            "iscrowd": 0
                        }
                        coco["annotations"].append(annotation_info)
                        annotation_id += 1
            # 写图像
            writeTiff(cropped, crop_geotrans, proj, os.path.join(SavePath, f"{new_name}.tif"))
            pbar.update(1)
            new_name += 1
        
        # 向前裁剪最后一行
        for j in range(int((width - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))):
            x_offset = int(j * CropSize * (1 - RepetitionRate))
            y_offset = height - CropSize
            cropped = dataset_img.ReadAsArray(x_offset, y_offset, CropSize, CropSize)
            
            XGeo, YGeo = CoordTransf(x_offset, y_offset, geotrans)
            crop_geotrans = (XGeo, geotrans[1], geotrans[2], YGeo, geotrans[4], geotrans[5])

            img_info = {
                "file_name": img_path,
                "height": CropSize,
                "width": CropSize,
                "id": new_name
            }
            coco['images'].append(img_info)

            # 获取当前裁剪图像的几何信息
            img_geom = box(XGeo, YGeo, XGeo + CropSize * geotrans[1], YGeo + CropSize * geotrans[5])
            img_geom = transform(lambda x, y: (x, y), img_geom)

            # 遍历shapefile,获取与当前裁剪图像相交的几何信息
            for idx, row in gdf.iterrows():
                geom = row.geometry
                if geom.intersects(img_geom):
                    intersection = geom.intersection(img_geom)
                    if not intersection.is_empty:
                        segmentation = [get_pixel_segmentation(intersection, geotrans, x_offset, y_offset)]
                        bbox = list(intersection.bounds)
                        area = intersection.area

                        annotation_info = {
                            "id": annotation_id,
                            "image_id": new_name,
                            "category_id": 1,
                            "segmentation": segmentation,
                            "area": area,
                            "bbox": bbox,
                            "iscrowd": 0
                        }
                        coco["annotations"].append(annotation_info)
                        annotation_id += 1

            writeTiff(cropped, crop_geotrans, proj, os.path.join(SavePath, f"{new_name}.tif"))
            pbar.update(1)
            # 文件名 + 1
            new_name += 1

        # 裁剪右下角
        x_offset = width - CropSize
        y_offset = height - CropSize
        cropped = dataset_img.ReadAsArray(x_offset, y_offset, CropSize, CropSize)
        
        XGeo, YGeo = CoordTransf(x_offset, y_offset, geotrans)
        crop_geotrans = (XGeo, geotrans[1], geotrans[2], YGeo, geotrans[4], geotrans[5])

        img_info = {
            "file_name": img_path,
            "height": CropSize,
            "width": CropSize,
            "id": new_name
        }
        coco['images'].append(img_info)

        # 获取当前裁剪图像的几何信息
        img_geom = box(XGeo, YGeo, XGeo + CropSize * geotrans[1], YGeo + CropSize * geotrans[5])
        img_geom = transform(lambda x, y: (x, y), img_geom)

        # 遍历shapefile,获取与当前裁剪图像相交的几何信息
        for idx, row in gdf.iterrows():
            geom = row.geometry
            if geom.intersects(img_geom):
                intersection = geom.intersection(img_geom)
                if not intersection.is_empty:
                    segmentation = [get_pixel_segmentation(intersection, geotrans, x_offset, y_offset)]
                    bbox = list(intersection.bounds)
                    area = intersection.area

                    annotation_info = {
                        "id": annotation_id,
                        "image_id": new_name,
                        "category_id": 1,
                        "segmentation": segmentation,
                        "area": area,
                        "bbox": bbox,
                        "iscrowd": 0
                    }
                    coco["annotations"].append(annotation_info)
                    annotation_id += 1

        writeTiff(cropped, crop_geotrans, proj, os.path.join(SavePath, f"{new_name}.tif"))
        pbar.update(1)
        new_name += 1

        # coco save path是tif的父目录，下的coco.json
        tifFatherPath = os.path.dirname(TifPath)
        coco_save_path = os.path.join(tifFatherPath, "coco.json")

    with open(coco_save_path, "w") as f:
        json.dump(coco, f)


if __name__ == "__main__":
    TifPath = '/mnt/data/Tree/Test/Four_CLUE_201804.tif'
    ShapefilePath = '/mnt/data/Tree/Test/Treecanopies_2018_cityofMelb_four_CLUE/Treecanopies_2018_cityofMelb_four_CLUE.shp'
    SavePath = "/mnt/data/Tree/Test/Four_CLUE_201804_512_02tiles/"
    CropSize = 512
    RepetitionRate = 0.2
    TifwithShape2COCO(TifPath, ShapefilePath, SavePath, CropSize, RepetitionRate)