import numpy as np
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
import os
import glob
from pathlib import Path

def load_and_filter_data(file_path):
    """
    加载数据并只保留有树木的地方（值大于0）
    """
    data = np.load(file_path)
    data = data[data <80]
    return data[data > 0]  # 只保留树高大于0的值

def calculate_wasserstein_distance(dist1, dist2):
    """
    计算两个分布之间的Wasserstein距离
    """
    return wasserstein_distance(dist1, dist2)

def plot_distributions(entire_area, train_data, val_data, test_data, save_path):
    """
    绘制分布对比图
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axs = plt.subplots(2, 1, figsize=(12, 12))

    # 计算Wasserstein距离
    train_dist = calculate_wasserstein_distance(entire_area, train_data)
    val_dist = calculate_wasserstein_distance(entire_area, val_data)
    test_dist = calculate_wasserstein_distance(entire_area, test_data)

    # 绘制直方图
    bins = np.linspace(0, max(np.max(entire_area), np.max(train_data), 
                             np.max(val_data), np.max(test_data)), 50)  # 50个bins
    
    axs[0].hist(entire_area, bins=bins, alpha=0.5, label='Entire Area', density=True, color='black')
    axs[0].hist(train_data, bins=bins, alpha=0.5, label=f'Train (W={train_dist:.2f})', density=True, color='blue')
    axs[0].hist(val_data, bins=bins, alpha=0.5, label=f'Val (W={val_dist:.2f})', density=True, color='green')
    axs[0].hist(test_data, bins=bins, alpha=0.5, label=f'Test (W={test_dist:.2f})', density=True, color='red')
    
    axs[0].set_title('Tree Height Distribution Comparison')
    axs[0].set_xlabel('Tree Height')
    axs[0].set_ylabel('Density')
    axs[0].legend()
    axs[0].grid(True)

    # 绘制ECDF
    def ecdf(data):
        x = np.sort(data)
        y = np.arange(1, len(data) + 1) / len(data)
        return x, y

    x_entire, y_entire = ecdf(entire_area)
    x_train, y_train = ecdf(train_data)
    x_val, y_val = ecdf(val_data)
    x_test, y_test = ecdf(test_data)

    axs[1].plot(x_entire, y_entire, label='Entire Area', color='black')
    axs[1].plot(x_train, y_train, label=f'Train (W={train_dist:.2f})', color='blue')
    axs[1].plot(x_val, y_val, label=f'Val (W={val_dist:.2f})', color='green')
    axs[1].plot(x_test, y_test, label=f'Test (W={test_dist:.2f})', color='red')

    axs[1].set_title('Empirical Cumulative Distribution Function')
    axs[1].set_xlabel('Tree Height')
    axs[1].set_ylabel('Cumulative Probability')
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    return train_dist, val_dist, test_dist

def main():
    # 设置随机种子
    np.random.seed(42)
    
    # 加载数据
    data_dir = "/mnt/data/TreeHeight/tree_height_data"
    entire_area = load_and_filter_data(os.path.join(data_dir, "entire_area_heights.npy"))
    
    # 获取所有grid文件
    grid_files = glob.glob(os.path.join(data_dir, "grid_*.npy"))
    n_grids = len(grid_files)
    
    # 计算每个集合的大小
    n_train = int(n_grids * 0.6)
    n_val = int(n_grids * 0.2)
    n_test = n_grids - n_train - n_val
    
    best_distances = float('inf')
    best_combination = None
    
    # 进行20次随机组合
    for i in range(100):
        # 随机打乱文件列表
        np.random.shuffle(grid_files)
        
        # 分割数据集
        train_files = grid_files[:n_train]
        val_files = grid_files[n_train:n_train + n_val]
        test_files = grid_files[n_train + n_val:]
        
        # 加载并合并数据
        train_data = np.concatenate([load_and_filter_data(f) for f in train_files])
        val_data = np.concatenate([load_and_filter_data(f) for f in val_files])
        test_data = np.concatenate([load_and_filter_data(f) for f in test_files])
        
        # 计算Wasserstein距离
        train_dist = calculate_wasserstein_distance(entire_area, train_data)
        val_dist = calculate_wasserstein_distance(entire_area, val_data)
        test_dist = calculate_wasserstein_distance(entire_area, test_data)
        
        # 计算总距离
        total_dist = train_dist + val_dist + test_dist
        
        # 更新最佳结果
        if total_dist < best_distances:
            best_distances = total_dist
            best_combination = (train_files, val_files, test_files)
            print(f"New best combination found (iteration {i+1}):")
            print(f"Train W-distance: {train_dist:.4f}")
            print(f"Val W-distance: {val_dist:.4f}")
            print(f"Test W-distance: {test_dist:.4f}")
            print(f"Total W-distance: {total_dist:.4f}")
            print("---")
            
            # 绘制最佳组合的分布图
            plot_distributions(
                entire_area, train_data, val_data, test_data,
                os.path.join(data_dir, "best_distribution_comparison.png")
            )
    
    # 保存最佳组合的文件列表
    train_files, val_files, test_files = best_combination
    np.save(os.path.join(data_dir, "best_train_files.npy"), train_files)
    np.save(os.path.join(data_dir, "best_val_files.npy"), val_files)
    np.save(os.path.join(data_dir, "best_test_files.npy"), test_files)

if __name__ == "__main__":
    main() 