import numpy as np
import os
import geopandas as gpd
import pandas as pd
from tqdm import tqdm

from matplotlib.patches import Polygon, Rectangle

from matplotlib import pyplot as plt


# 对树的shapefile进行预处理，获取单独的树
def get_single_tree(shapefile_path, save_path=None):
    """
    获取单独的树
    :param shapefile_path: shapefile路径
    :param save_path: 保存路径
    :return:
    """
    gdf = gpd.read_file(shapefile_path)
    gdf_out = gpd.GeoDataFrame(geometry=gpd.GeoSeries())

    # 处理投影系使得gdf_out与gdf一致
    gdf_out.set_crs(gdf.crs, inplace=True)

    geom_type_list = []

    with tqdm(total=gdf.shape[0]) as pbar:
        for index, row in gdf.iterrows():
            geom_type = row.geometry.geom_type
            if geom_type not in geom_type_list:
                geom_type_list.append(geom_type)
            area = row.geometry.area
            
            # 长宽比
            box = row.geometry.bounds  # box的单位是什么？答案是：与坐标系一致
            ratio = (box[2] - box[0]) / (box[3] - box[1])

            # if index == 0:
            #     # 画图
            #     fig, ax = plt.subplots()
            #     ax.set_aspect('equal')
            #     ax.set_xlim([box[0] - 1, box[2] + 1])  # 扩大视图边界
            #     ax.set_ylim([box[1] - 1, box[3] + 1])

            #     # 画 Polygon
            #     coords_2d = np.array(row.geometry.exterior.coords)[:, :2]
            #     polygon_patch = Polygon(coords_2d, facecolor='blue', alpha=0.5)
            #     ax.add_patch(polygon_patch)
                
            #     # 画 bounding box
            #     rect = Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], 
            #                     edgecolor='red', facecolor='none', linewidth=2)
            #     ax.add_patch(rect)

            #     plt.savefig('test.png')
            #     plt.close()
            #     break

            # 面积占比
            ratio_area = area / (box[2] - box[0]) / (box[3] - box[1])

            # 保留条件：面积小于100，长宽比小于2，面积占比大于0.5，没有孔洞，不是MultiPolygon
            if geom_type!='MultiPolygon':
                # 是否有孔洞
                if len(row.geometry.interiors) == 0:
                    if area < 200 and ratio < 2 and ratio_area > 0.6:
                        # 使用concat方法添加新行
                        gdf_out = gpd.GeoDataFrame(pd.concat([gdf_out, gpd.GeoDataFrame(geometry=[row.geometry])], ignore_index=True))
            pbar.update(1)
    if save_path is not None:
        gdf_out.to_file(save_path)
        print('Save to', save_path)
    print(geom_type_list)



if __name__ == '__main__':
    get_single_tree('/mnt/data/Tree/Test/Treecanopies_2018_cityofMelb_four_CLUE/Treecanopies_2018_cityofMelb_four_CLUE.shp', save_path='/mnt/data/Tree/Test/Treecanopies_2018_cityofMelb_four_CLUE/Treecanopies_2018_cityofMelb_four_CLUE_single_tree.shp')