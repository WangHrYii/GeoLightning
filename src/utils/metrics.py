import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from scipy.stats import gaussian_kde

def read_tif(file_path):
    """使用GDAL读取单波段TIFF文件"""
    dataset = gdal.Open(file_path)
    if dataset is None:
        raise ValueError(f"无法打开文件：{file_path}")
    band = dataset.GetRasterBand(1)
    return band.ReadAsArray()


def read_mask_tif(file_path, unique_value):
    """考虑到mask标注可能会有奇怪的标注形式，这里按照unique_value来处理"""
    dataset = gdal.Open(file_path)
    if dataset is None:
        raise ValueError(f"无法打开文件：{file_path}")
    band = dataset.GetRasterBand(1)
    data = band.ReadAsArray()
    mask = np.zeros_like(data)

    in_value = 1
    for value in unique_value:
        mask[data == value] = in_value
        in_value += 1
    
    return mask
    


def calculate_metrics(y_true, y_pred):
    """计算回归指标"""
    return {
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred)
    }

def plot_density_scatter(y_true, y_pred, metrics, title, filename):
    """绘制密度着色散点图（限制坐标轴范围）"""
    plt.figure(figsize=(10, 8))
    
    # 随机采样10000个点（或全部数据如果不足）
    n_samples = min(100000, len(y_true))
    sample_idx = np.random.choice(len(y_true), n_samples, replace=False)
    y_true_sample = y_true[sample_idx]
    y_pred_sample = y_pred[sample_idx]

    # 计算密度
    try:
        xy = np.vstack([y_true_sample, y_pred_sample])
        z = gaussian_kde(xy)(xy)
    except:
        z = np.ones_like(y_true_sample)  # 密度计算失败时使用统一颜色

    # 绘制密度散点图
    scatter = plt.scatter(
        y_true_sample, y_pred_sample,
        c=z, cmap='viridis',
        s=8, alpha=0.7,
        edgecolors='none'
    )
    plt.colorbar(scatter, label='Density', shrink=0.8)


    # 添加趋势线（限制在0-60范围内）
    if len(y_true) > 1:
        lr = LinearRegression().fit(y_true.reshape(-1,1), y_pred)
        xlim = np.linspace(0, 60, 100)  # 强制从0到60生成趋势线
        plt.plot(xlim, lr.predict(xlim.reshape(-1,1)), 
                'r--', linewidth=2, label='Trend line')
        
    # 添加1:1线
    plt.plot([0, 60], [0, 60], 'k-', linewidth=1.5, label='1:1 line', alpha=0.7)

    # 添加指标文本和图例
    text = "\n".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
    plt.text(0.05, 0.95, text + "\n", transform=plt.gca().transAxes,
            fontsize=12, verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.8))
    
    # 设置坐标轴范围
    plt.xlim(0, 60)
    plt.ylim(0, 60)
    
    plt.title(f"{title}\nDensity Scatter Plot (0-60m)", fontsize=14)
    plt.xlabel("True Height (m)", fontsize=12)
    plt.ylabel("Predicted Height (m)", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 在指标文本下方添加图例
    plt.legend(loc='upper left', bbox_to_anchor=(0.03, 0.82), 
                       fontsize=12, framealpha=0.8)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()



# 主程序
if __name__ == "__main__":
    # 读取数据
    true_height = read_tif('/mnt/data/Tree/TreeHeight/nDSM_cropped_boundary.tif').astype(float)
    # true_mask = read_tif('/mnt/data/Tree/TreeHeight/tree_cover/tree_cropped_boundary.tif').astype(float)
    true_mask = read_mask_tif('/mnt/data/Tree/TreeHeight/tree_cover/tree_crop_bound_reproj.tif', [101])
    pred_height = read_tif('/mnt/data/Tree/TreeHeight/tree_cover/prediction_height.tif').astype(float)

    true_height_shape = true_height.shape
    pred_height_shape = pred_height.shape
    true_mask_shape = true_mask.shape

    # 因为矢量裁剪可能会导致一两个像素的偏差，这里选择shape的最小值,然后都按照最小值范围
    shapes = [true_height_shape, pred_height_shape, true_mask_shape]
    min_rows = min(s[0] for s in shapes)
    min_cols = min(s[1] for s in shapes)
    min_shape = (min_rows, min_cols)

    # 裁剪所有数据到统一尺寸
    true_height = true_height[:min_rows, :min_cols]
    pred_height = pred_height[:min_rows, :min_cols]
    true_mask = true_mask[:min_rows, :min_cols]

    

    # 案例1计算：全域比较（真值非树木区置零）
    y_true_case1 = (true_height * true_mask).ravel()  # ravle()将多维数组展平
    y_pred_case1 = pred_height.ravel()
    metrics_case1 = calculate_metrics(y_true_case1, y_pred_case1)

    # 案例2计算：仅树木区域
    tree_mask = true_mask == 1
    y_true_case2 = true_height[tree_mask]
    y_pred_case2 = pred_height[tree_mask]
    metrics_case2 = calculate_metrics(y_true_case2, y_pred_case2)

    # 生成可视化
    plot_density_scatter(y_true_case1, y_pred_case1, metrics_case1,
                        "Case 1: Full Area Comparison", 
                        "case1_density_scatter.png")
    
    plot_density_scatter(y_true_case2, y_pred_case2, metrics_case2,
                        "Case 2: True Tree Areas Only", 
                        "case2_density_scatter.png")

    # 打印结果
    print("Case 1 Metrics:")
    print("\n".join([f"{k}: {v:.4f}" for k, v in metrics_case1.items()]))
    print("\nCase 2 Metrics:")
    print("\n".join([f"{k}: {v:.4f}" for k, v in metrics_case2.items()]))
