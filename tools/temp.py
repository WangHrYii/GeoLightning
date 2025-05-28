import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

train_files = np.load("/mnt/data/TreeHeight/tree_height_data/best_train_files.npy")
val_files = np.load("/mnt/data/TreeHeight/tree_height_data/best_val_files.npy")
test_files = np.load("/mnt/data/TreeHeight/tree_height_data/best_test_files.npy")

from analyze_distributions import load_and_filter_data


def toSHP(train_files, val_files, test_files):
    train_ids = [int(file.split("/")[-1].split(".")[0].split("_")[1]) for file in train_files]
    val_ids = [int(file.split("/")[-1].split(".")[0].split("_")[1]) for file in val_files]
    test_ids = [int(file.split("/")[-1].split(".")[0].split("_")[1]) for file in test_files]

    shapefile_path = "/mnt/data/TreeHeight/dc_reprojected.shp"

    gdf = gpd.read_file(shapefile_path)

    print(gdf.columns)

    gdf_train = gdf[gdf["OBJECTID_1"].isin(train_ids)]
    gdf_train.to_file("/mnt/data/TreeHeight/train_data.shp", driver="ESRI Shapefile")

    gdf_val = gdf[gdf["OBJECTID_1"].isin(val_ids)]
    gdf_val.to_file("/mnt/data/TreeHeight/val_data.shp", driver="ESRI Shapefile")

    gdf_test = gdf[gdf["OBJECTID_1"].isin(test_ids)]
    gdf_test.to_file("/mnt/data/TreeHeight/test_data.shp", driver="ESRI Shapefile")


def get_histograms(train_files, val_files, test_files):
    train_data = np.concatenate([load_and_filter_data(file) for file in train_files])
    val_data = np.concatenate([load_and_filter_data(file) for file in val_files])
    test_data = np.concatenate([load_and_filter_data(file) for file in test_files])

    print(f"Data shapes - Train: {train_data.shape}, Val: {val_data.shape}, Test: {test_data.shape}")

    # 计算直方图密度数据
    bins = 100
    
    # 使用统一的bin范围确保可比性
    all_data = np.concatenate([train_data, val_data, test_data])
    data_min, data_max = all_data.min(), all_data.max()
    bin_edges = np.linspace(data_min, data_max, bins + 1)
    
    # 计算每个数据集的直方图
    train_counts, _ = np.histogram(train_data, bins=bin_edges, density=True)
    val_counts, _ = np.histogram(val_data, bins=bin_edges, density=True)
    test_counts, _ = np.histogram(test_data, bins=bin_edges, density=True)
    
    # 计算bin中心点
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # 计算统计信息
    train_mean, train_std = np.mean(train_data), np.std(train_data)
    val_mean, val_std = np.mean(val_data), np.std(val_data)
    test_mean, test_std = np.mean(test_data), np.std(test_data)
    
    # 计算分布之间的差异
    train_val_diff = train_counts - val_counts
    train_test_diff = train_counts - test_counts
    val_test_diff = val_counts - test_counts
    
    # 为重合分布提供偏移量
    max_density = max(train_counts.max(), val_counts.max(), test_counts.max())
    offset_factor = max_density * 0.15
    
    # 创建带偏移的密度数据
    train_counts_offset = train_counts
    val_counts_offset = val_counts + offset_factor
    test_counts_offset = test_counts + offset_factor * 2
    
    # 计算累积密度
    cumulative_train = np.cumsum(train_counts) / np.sum(train_counts)
    cumulative_val = np.cumsum(val_counts) / np.sum(val_counts)
    cumulative_test = np.cumsum(test_counts) / np.sum(test_counts)
    
    # 创建可视化
    plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. 原始重合分布
    ax1.plot(bin_centers, train_counts, label='Train', linewidth=2, alpha=0.8)
    ax1.plot(bin_centers, val_counts, label='Val', linewidth=2, alpha=0.8)
    ax1.plot(bin_centers, test_counts, label='Test', linewidth=2, alpha=0.8)
    ax1.set_title('Original Overlapping Distributions', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Density')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 累积分布函数
    ax2.plot(bin_centers, cumulative_train, label='Train CDF', linewidth=2)
    ax2.plot(bin_centers, cumulative_val, label='Val CDF', linewidth=2)
    ax2.plot(bin_centers, cumulative_test, label='Test CDF', linewidth=2)
    ax2.set_title('Cumulative Distribution Functions', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Value')
    ax2.set_ylabel('Cumulative Probability')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("/home/whr/Codes/GeoLightning/visualizations/enhanced_distributions.png", 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # 保存数据
    density_data = {
        'basic_data': {
            'bin_centers': bin_centers,
            'bin_edges': bin_edges,
            'train_density': train_counts,
            'val_density': val_counts,
            'test_density': test_counts
        },
        'statistics': {
            'train': {'mean': train_mean, 'std': train_std, 'count': len(train_data)},
            'val': {'mean': val_mean, 'std': val_std, 'count': len(val_data)},
            'test': {'mean': test_mean, 'std': test_std, 'count': len(test_data)}
        },
        'visualization_data': {
            'offset_densities': {
                'train': train_counts_offset,
                'val': val_counts_offset,
                'test': test_counts_offset
            },
            'cumulative_densities': {
                'train': cumulative_train,
                'val': cumulative_val,
                'test': cumulative_test
            },
            'differences': {
                'train_val_diff': train_val_diff,
                'train_test_diff': train_test_diff,
                'val_test_diff': val_test_diff
            }
        }
    }
    
    np.save("/home/whr/Codes/GeoLightning/visualizations/density_data.npy", density_data)
    
    print(f"Visualization completed!")
    print(f"- Charts saved: enhanced_distributions.png")
    print(f"- Data saved: density_data.npy")



if __name__ == "__main__":
    # toSHP(train_files, val_files, test_files)
    get_histograms(train_files, val_files, test_files)