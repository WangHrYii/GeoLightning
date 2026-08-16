import os
import rasterio
import numpy as np
import logging

def replace_invalid_values(data, nodata_values):
    """
    将数据中的Nodata值和NaN值替换为0
    """
    for nodata in nodata_values:
        data[data == nodata] = 0
    data[np.isnan(data)] = 0
    return data


def load_image(filename):
    # 使用rasterio读取图像，返回ndarray, 要求支持任意波段的图像
    with rasterio.open(filename) as src:
        img = src.read()
        return img


# 切分数据集为训练集、验证集和测试集
def splitData(data_dir, label_dirs=[], ratio = None, mode = 'random', data_suffix = '.tif', rm_nodata = True):
    """数据集切分函数，label_dirs为标签数据集路径，按照data_dir中的数据集对label_dirs中的数据集进行相同切分
    data_dir: 数据集路径（输入）
    label_dirs: 标签数据集路径（输出），支持多个标签数据集
    ratio: 训练集、验证集和测试集的比例，形式为[ratio_train, ratio_val, ratio_test]，支持两种模式
    mode: 切分模式，random为随机切分，row为按行数切分
    data_suffix: 数据集后缀
    rm_nodata: 是否删除无效数据
    TODO: 1.支持按照列数切分数据集
          2.支持按照行数和列数切分数据集
          3.支持按照数据集的大小切分数据集
          4.支持label_dirs中每个标签数据集的data_suffix可输入列表
          5.支持不该动原始数据，而是按照输入的名字生成新的数据集，所有的数据集都在同一个文件夹下
          6.容错率进一步提高，目标是能够处理任意数据集，可以直接用于产品生产
    """
    if mode == 'random':
        # 随机切分数据集
        if ratio is None:
            ratio = [0.8, 0.1, 0.1]
        assert sum(ratio) == 1
        data = os.listdir(data_dir)
        n = len(data)
        n_train = int(ratio[0] * n)
        n_val = int(ratio[1] * n)
        n_test = n - n_train - n_val
        data_train = data[:n_train]
        data_val = data[n_train:n_train+n_val]
        data_test = data[n_train+n_val:]
        os.mkdir(os.path.join(data_dir, 'train'))
        os.mkdir(os.path.join(data_dir, 'val'))
        os.mkdir(os.path.join(data_dir, 'test'))
        for i in data_train:
            os.rename(os.path.join(data_dir, i), os.path.join(data_dir, 'train', i))
        for i in data_val:
            os.rename(os.path.join(data_dir, i), os.path.join(data_dir, 'val', i))
        for i in data_test:
            os.rename(os.path.join(data_dir, i), os.path.join(data_dir, 'test', i))

    elif mode == 'row':
        # 按行数切分数据集，数据名称按照文件名排序，形式为{row}_{col}.xxx
        # 先计算数据集的行数和列数
        data = os.listdir(data_dir)
        # 去除非数据文件
        data = [i for i in data if i.endswith(data_suffix)]
        n = len(data)
        row = 0
        col = 0
        for i in data:
            row = max(row, int(i.split('_')[0]))
            col = max(col, int(i.split('_')[1].split('.')[0]))
        
        print(f'row: {row}, col: {col}')
        
        if len(ratio) == 2:  # 只区分训练集和验证集
            assert sum(ratio) == 1, f'Sum of ratio should be 1, but got {sum(ratio)}, and ratio is {ratio[0]}, {ratio[1]}'

            row_split = int(row * ratio[0])  # 训练集的行数

            # 创建train和val文件夹
            if not os.path.exists(os.path.join(data_dir, 'train')):
                os.mkdir(os.path.join(data_dir, 'train'))
            if not os.path.exists(os.path.join(data_dir, 'val')):
                os.mkdir(os.path.join(data_dir, 'val'))
            
            # 在label_dirs中创建train和val文件夹
            if len(label_dirs) > 0:
                for label_dir in label_dirs:
                    if not os.path.exists(os.path.join(label_dir, 'train')):
                        os.mkdir(os.path.join(label_dir, 'train'))
                    if not os.path.exists(os.path.join(label_dir, 'val')):
                        os.mkdir(os.path.join(label_dir, 'val'))

            # 处理训练集
            for i in range(row_split):
                for j in range(col+1):
                    data_path = os.path.join(data_dir, f'{i}_{j}{data_suffix}')
                    if os.path.exists(data_path):
                        # 判断是否是有效数据
                        if rm_nodata:
                            data_name = data_path.split('/')[-1]
                            data = load_image(data_path)
                            # 预处理数据，将Nodata和NaN的替换为0
                            data = replace_invalid_values(data, [np.nan, -9999])
                            if data.max() == 0:
                                os.remove(data_path)
                                for label_dir in label_dirs:
                                    os.remove(os.path.join(label_dir, data_name))
                                continue  # 跳过无效数据

                        os.rename(data_path, os.path.join(data_dir, 'train', f'{i}_{j}{data_suffix}'))
                        for label_dir in label_dirs:
                            os.rename(os.path.join(label_dir, f'{i}_{j}{data_suffix}'), os.path.join(label_dir, 'train', f'{i}_{j}{data_suffix}'))
            
            # 处理验证集
            for i in range(row_split, row+1):
                for j in range(col+1):
                    data_path = os.path.join(data_dir, f'{i}_{j}{data_suffix}')
                    if os.path.exists(data_path):
                        # 判断是否是有效数据
                        if rm_nodata:
                            data_name = data_path.split('/')[-1]
                            data = load_image(data_path)
                            # 预处理数据，将Nodata和NaN的替换为0
                            data = replace_invalid_values(data, [np.nan, -9999])
                            if data.max() == 0:
                                os.remove(data_path)
                                for label_dir in label_dirs:
                                    os.remove(os.path.join(label_dir, data_name))
                                continue
                        os.rename(data_path, os.path.join(data_dir, 'val', f'{i}_{j}{data_suffix}'))
                        for label_dir in label_dirs:
                            os.rename(os.path.join(label_dir, f'{i}_{j}{data_suffix}'), os.path.join(label_dir, 'val', f'{i}_{j}{data_suffix}'))
            
            # 检查data_dir中的train, val和label_dir中的train, val是否一致
            num_train = len(os.listdir(os.path.join(data_dir, 'train')))
            num_val = len(os.listdir(os.path.join(data_dir, 'val')))
            print(f'num_train: {num_train}, num_val: {num_val}')
            for label_dir in label_dirs:
                label_train = os.listdir(os.path.join(label_dir, 'train'))
                label_val = os.listdir(os.path.join(label_dir, 'val'))
                assert len(label_train) == num_train, f'Number of train data in {label_dir} is not equal to {num_train}'
                logging.info(f'Number of train data in {label_dir} is equal to {num_train}')
                assert len(label_val) == num_val, f'Number of val data in {label_dir} is not equal to {num_val}'
                logging.info(f'Number of val data in {label_dir} is equal to {num_val}')
                logging.info(f'Data Splitting is done with mode: {mode}, ratio: {ratio}, num_train: {num_train}, num_val: {num_val}')

        if len(ratio) == 3:  # 区分训练集、验证集和测试集
            row_split = int(row * ratio[0])                    # 训练集的行数
            row_split_val = int(row * (ratio[0] + ratio[1]))   # 验证集的行数

            # 输入数据创建train、val和test文件夹
            if not os.path.exists(os.path.join(data_dir, 'train')):
                os.mkdir(os.path.join(data_dir, 'train'))
            if not os.path.exists(os.path.join(data_dir, 'val')):
                os.mkdir(os.path.join(data_dir, 'val'))
            if not os.path.exists(os.path.join(data_dir, 'test')):
                os.mkdir(os.path.join(data_dir, 'test'))
            
            # 在label_dirs中创建train、val和test文件夹
            if len(label_dirs) > 0:
                for label_dir in label_dirs:
                    if not os.path.exists(os.path.join(label_dir, 'train')):
                        os.mkdir(os.path.join(label_dir, 'train'))
                    if not os.path.exists(os.path.join(label_dir, 'val')):
                        os.mkdir(os.path.join(label_dir, 'val'))
                    if not os.path.exists(os.path.join(label_dir, 'test')):
                        os.mkdir(os.path.join(label_dir, 'test'))

            # 处理训练集
            for i in range(row_split):
                for j in range(col+1):
                    data_path = os.path.join(data_dir, f'{i}_{j}{data_suffix}')
                    if os.path.exists(data_path):
                        # 判断是否是有效数据
                        if rm_nodata:
                            data_name = data_path.split('/')[-1]
                            data = load_image(data_path)
                            # 预处理数据，将Nodata和NaN的替换为0
                            data = replace_invalid_values(data, [np.nan, -9999])
                            if data.max() == 0:
                                os.remove(data_path)
                                for label_dir in label_dirs:
                                    os.remove(os.path.join(label_dir, data_name))
                                continue
                        os.rename(data_path, os.path.join(data_dir, 'train', f'{i}_{j}{data_suffix}'))
                        for label_dir in label_dirs:
                            os.rename(os.path.join(label_dir, f'{i}_{j}{data_suffix}'), os.path.join(label_dir, 'train', f'{i}_{j}{data_suffix}'))

            # 处理验证集
            for i in range(row_split, row_split_val):
                for j in range(col+1):
                    data_path = os.path.join(data_dir, f'{i}_{j}{data_suffix}')
                    if os.path.exists(data_path):
                        # 判断是否是有效数据
                        if rm_nodata:
                            data_name = data_path.split('/')[-1]
                            data = load_image(data_path)
                            # 预处理数据，将Nodata和NaN的替换为0
                            data = replace_invalid_values(data, [np.nan, -9999])
                            if data.max() == 0:
                                os.remove(data_path)
                                for label_dir in label_dirs:
                                    os.remove(os.path.join(label_dir, data_name))
                                continue
                        os.rename(data_path, os.path.join(data_dir, 'val', f'{i}_{j}{data_suffix}'))
                        for label_dir in label_dirs:
                            os.rename(os.path.join(label_dir, f'{i}_{j}{data_suffix}'), os.path.join(label_dir, 'val', f'{i}_{j}{data_suffix}'))

            # 处理测试集
            for i in range(row_split_val, row+1):
                for j in range(col+1):
                    data_path = os.path.join(data_dir, f'{i}_{j}{data_suffix}')
                    if os.path.exists(data_path):
                        # 判断是否是有效数据
                        if rm_nodata:
                            data_name = data_path.split('/')[-1]
                            data = load_image(data_path)
                            # 预处理数据，将Nodata和NaN的替换为0
                            data = replace_invalid_values(data, [np.nan, -9999])
                            if data.max() == 0:
                                os.remove(data_path)
                                for label_dir in label_dirs:
                                    os.remove(os.path.join(label_dir, data_name))
                                continue
                        os.rename(data_path, os.path.join(data_dir, 'test', f'{i}_{j}{data_suffix}'))
                        for label_dir in label_dirs:
                            os.rename(os.path.join(label_dir, f'{i}_{j}{data_suffix}'), os.path.join(label_dir, 'test', f'{i}_{j}{data_suffix}'))

            # 检查data_dir中的train, val, test和label_dir中的train, val, test是否一致
            num_train = len(os.listdir(os.path.join(data_dir, 'train')))
            num_val = len(os.listdir(os.path.join(data_dir, 'val')))
            num_test = len(os.listdir(os.path.join(data_dir, 'test')))

            for label_dir in label_dirs:
                label_train = os.listdir(os.path.join(label_dir, 'train'))
                label_val = os.listdir(os.path.join(label_dir, 'val'))
                label_test = os.listdir(os.path.join(label_dir, 'test'))
                assert len(label_train) == num_train, f'Number of train data in {label_dir} is not equal to {num_train}'
                logging.info(f'Number of train data in {label_dir} is equal to {num_train}')

                assert len(label_val) == num_val, f'Number of val data in {label_dir} is not equal to {num_val}'
                logging.info(f'Number of val data in {label_dir} is equal to {num_val}')

                assert len(label_test) == num_test, f'Number of test data in {label_dir} is not equal to {num_test}'
                logging.info(f'Number of test data in {label_dir} is equal to {num_test}')
                logging.info(f'Data Splitting is done with mode: {mode}, ratio: {ratio}, num_train: {num_train}, num_val: {num_val}, num_test: {num_test}')



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Split aligned raster datasets')
    parser.add_argument('data_dir')
    parser.add_argument('label_dirs', nargs='+')
    args = parser.parse_args()
    splitData(
        args.data_dir,
        label_dirs=args.label_dirs,
        ratio=[0.6, 0.4],
        mode='row',
        data_suffix='.tif',
    )
