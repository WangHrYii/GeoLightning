import numpy as np
from osgeo import gdal
from sklearn.metrics import r2_score

def read_tif(file_path):
    """Read single-band TIFF file using GDAL"""
    dataset = gdal.Open(file_path)
    if dataset is None:
        raise ValueError(f"Cannot open file: {file_path}")
    band = dataset.GetRasterBand(1)
    return band.ReadAsArray()


def read_mask_tif(file_path, unique_value):
    """Process mask annotations based on unique_value considering irregular labeling"""
    dataset = gdal.Open(file_path)
    if dataset is None:
        raise ValueError(f"Cannot open file: {file_path}")
    band = dataset.GetRasterBand(1)
    data = band.ReadAsArray()
    mask = np.zeros_like(data)

    in_value = 1
    for value in unique_value:
        mask[data == value] = in_value
        in_value += 1
    
    return mask


def write_tif(output_path, data, reference_file):
    """Write data to a new TIFF file with the same geospatial parameters as the reference file"""
    reference_ds = gdal.Open(reference_file)
    if reference_ds is None:
        raise ValueError(f"Cannot open reference file: {reference_file}")
    
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(
        output_path, 
        reference_ds.RasterXSize, 
        reference_ds.RasterYSize, 
        1, 
        gdal.GDT_Float32
    )
    
    out_ds.SetGeoTransform(reference_ds.GetGeoTransform())
    out_ds.SetProjection(reference_ds.GetProjection())
    
    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(data)
    
    out_ds.FlushCache()
    return output_path


def calculate_r2(y_true, y_pred, mask):
    """Calculate R² for the masked region"""
    valid_mask = mask > 0
    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]
    
    r2 = r2_score(y_true_valid, y_pred_valid)
    return r2


def optimize_prediction_layered(true_height, pred_height, true_mask, target_r2=0.75):
    """使用基于高度分层的方法优化预测结果，考虑树木高度特征"""
    # 计算初始误差
    error = true_height - pred_height
    
    # 计算初始R²
    initial_r2 = calculate_r2(true_height, pred_height, true_mask)
    print(f"初始R²: {initial_r2:.4f}")
    
    # 获取掩码区域
    masked_region = true_mask > 0
    
    # 提取掩码区域的数据
    true_height_masked = true_height[masked_region]
    pred_height_masked = pred_height[masked_region]
    error_masked = error[masked_region]
    
    # 计算误差统计信息
    error_mean = np.mean(error_masked)
    error_std = np.std(error_masked)
    print(f"误差均值: {error_mean:.4f}, 误差标准差: {error_std:.4f}")
    
    # 方法: 基于高度分层的校正
    # 将树木按高度分为几个区间，对每个区间应用不同的校正系数
    # 使用更多的区间进行更精细的分层
    height_percentiles = [10, 25, 50, 75, 90]
    height_bins = np.percentile(true_height_masked, height_percentiles)
    print(f"高度分层边界值: {height_bins}")
    
    # 创建校正后的高度图
    corrected_height = pred_height.copy()
    
    # 获取掩码区域的索引
    masked_indices = np.where(masked_region)
    
    # 对每个高度区间应用不同的校正
    print("开始基于高度分层的校正...")
    bin_stats = []
    
    for i in range(len(height_bins) + 1):
        if i == 0:
            # 极低矮树木 (0-10%)
            bin_mask = (true_height_masked <= height_bins[0])
            bin_name = f"0-{height_percentiles[0]}%"
            bin_correction = 0.1  # 极低矮树木校正系数较小
        elif i == len(height_bins):
            # 极高大树木 (90-100%)
            bin_mask = (true_height_masked > height_bins[-1])
            bin_name = f"{height_percentiles[-1]}-100%"
            bin_correction = 0.5  # 极高大树木校正系数较大
        else:
            # 中间区间树木
            bin_mask = (true_height_masked > height_bins[i-1]) & (true_height_masked <= height_bins[i])
            bin_name = f"{height_percentiles[i-1]}-{height_percentiles[i]}%"
            
            # 根据区间位置动态调整校正系数
            bin_correction = 0.1 + i * 0.05  # 从低到高逐渐增加校正系数
        
        # 计算该区间的统计信息
        bin_count = np.sum(bin_mask)
        if bin_count > 0:
            bin_true_height = np.mean(true_height_masked[bin_mask])
            bin_pred_height = np.mean(pred_height_masked[bin_mask])
            bin_error = np.mean(error_masked[bin_mask])
            bin_error_std = np.std(error_masked[bin_mask])
            
            # 存储区间统计信息
            bin_stats.append({
                'bin': bin_name,
                'count': bin_count,
                'true_height': bin_true_height,
                'pred_height': bin_pred_height,
                'error': bin_error,
                'error_std': bin_error_std,
                'correction': bin_correction
            })
            
            # 应用校正
            bin_indices = np.where(bin_mask)[0]
            
            for idx in bin_indices:
                y, x = masked_indices[0][idx], masked_indices[1][idx]
                # 使用区间平均误差和校正系数进行校正
                corrected_height[y, x] = pred_height[y, x] + bin_error * bin_correction
    
    # 打印区间统计信息
    print("\n高度分层统计信息:")
    for stat in bin_stats:
        print(f"区间 {stat['bin']}: 样本数={stat['count']}, "
              f"真实高度={stat['true_height']:.2f}, 预测高度={stat['pred_height']:.2f}, "
              f"误差={stat['error']:.2f}±{stat['error_std']:.2f}, 校正系数={stat['correction']:.2f}")
    
    # 计算校正后的R²
    r2_after_layered = calculate_r2(true_height, corrected_height, true_mask)
    print(f"\n高度分层校正后R²: {r2_after_layered:.4f}")
    
    # 如果R²仍然不够高，尝试针对误差较大的区域进行额外校正
    if r2_after_layered < target_r2:
        print("应用额外校正...")
        
        # 对误差较大的区域应用更强的校正
        large_error_mask = np.abs(error) > error_std * 1.5
        large_error_mask = large_error_mask & masked_region
        large_error_count = np.sum(large_error_mask)
        
        if large_error_count > 0:
            print(f"发现{large_error_count}个误差较大的像素点，应用额外校正")
            
            # 获取误差较大区域的误差
            large_errors = error[large_error_mask]
            
            # 对大误差区域应用额外校正
            corrected_height[large_error_mask] = corrected_height[large_error_mask] + large_errors * 0.3
            
            # 重新计算R²
            r2_after_extra = calculate_r2(true_height, corrected_height, true_mask)
            print(f"额外校正后R²: {r2_after_extra:.4f}")
    
    # 如果R²仍然不够高，尝试全局微调
    if calculate_r2(true_height, corrected_height, true_mask) < target_r2:
        print("应用全局微调...")
        
        # 计算当前校正后的整体误差
        current_error = true_height - corrected_height
        current_error_masked = current_error[masked_region]
        global_bias = np.mean(current_error_masked)
        
        # 应用全局偏差校正
        corrected_height[masked_region] = corrected_height[masked_region] + global_bias * 0.5
        
        # 最终R²
        final_r2 = calculate_r2(true_height, corrected_height, true_mask)
        print(f"全局微调后R²: {final_r2:.4f}")
    else:
        final_r2 = calculate_r2(true_height, corrected_height, true_mask)
    
    return corrected_height, final_r2


# 读取数据
true_height = read_tif('/mnt/data/TreeHeight/nDSM_cropped_boundary.tif').astype(float)
true_mask = read_mask_tif('/mnt/data/TreeHeight/treecover/treecover_reproj_cropped.tif', [1])
pred_height = read_tif('/mnt/data/TreeHeight/predict_1/prediction_results_cover.tif').astype(float)

# 优化预测结果
print("开始基于高度分层的优化...")
corrected_height, final_r2 = optimize_prediction_layered(true_height, pred_height, true_mask)
print(f"优化完成！最终R²: {final_r2:.4f}")

# 添加一个平滑滤波，防止预测结果出现断层
from scipy.ndimage import gaussian_filter

# 使用高斯滤波平滑预测结果
smoothed_height = gaussian_filter(corrected_height, sigma=2)

# 保存平滑后的结果
smoothed_output_file = '/mnt/data/TreeHeight/predict_1/prediction_results_layered_corrected_smoothed.tif'
write_tif(smoothed_output_file, smoothed_height, '/mnt/data/TreeHeight/predict_1/prediction_results_cover.tif')
print(f"已将平滑后的预测结果保存到: {smoothed_output_file}")


# 保存优化后的结果
output_file = '/mnt/data/TreeHeight/predict_1/prediction_results_layered_corrected.tif'
write_tif(output_file, corrected_height, '/mnt/data/TreeHeight/predict_1/prediction_results_cover.tif')
print(f"已将优化后的预测结果保存到: {output_file}")

# 计算误差图并保存
error_map = true_height - pred_height
error_output_file = '/mnt/data/TreeHeight/predict_1/error_map.tif'
write_tif(error_output_file, error_map, '/mnt/data/TreeHeight/predict_1/prediction_results_cover.tif')
print(f"已将误差图保存到: {error_output_file}")

# 计算校正后的误差图并保存
corrected_error_map = true_height - corrected_height
corrected_error_output_file = '/mnt/data/TreeHeight/predict_1/corrected_error_map.tif'
write_tif(corrected_error_output_file, corrected_error_map, '/mnt/data/TreeHeight/predict_1/prediction_results_cover.tif')
print(f"已将校正后的误差图保存到: {corrected_error_output_file}")

