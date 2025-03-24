# 实现各种各样的直方图
import numpy as np
import os
import geopandas as gpd

from matplotlib import pyplot as plt
from matplotlib import colors as mcolors


def plot_histogram(data, bins=10, title=None, xlabel=None, ylabel=None, save_path=None):
    """
    绘制直方图
    :param data: 数据
    :param bins: 直方图的柱数
    :param title: 标题
    :param xlabel: x轴标签
    :param ylabel: y轴标签
    :param save_path: 保存路径
    :return:
    """
    fig, ax = plt.subplots()
    ax.hist(data, bins=bins, color='blue', alpha=0.7, rwidth=0.85)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if save_path is not None:
        plt.savefig(save_path)



def shapefile_area_histogram(shapefile_path, save_path=None):
    """
    绘制shapefile中面积的直方图
    :param shapefile_path: shapefile路径
    :return:
    """
    gdf = gpd.read_file(shapefile_path)
    areas = gdf.area
    # 删除面积大于10000
    areas = areas[areas < 50]
    plot_histogram(areas, bins=80, title='Area Histogram', xlabel='Area', ylabel='Frequency', save_path=save_path)


if __name__ == '__main__':
    shapefile_area_histogram('/mnt/data/Tree/Test/Treecanopies_2018_cityofMelb_four_CLUE/Treecanopies_2018_cityofMelb_four_CLUE.shp', save_path='Treecanopies_2018_cityofMelb_four_CLUE_area_histogram.png')