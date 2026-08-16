import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from osgeo import gdal
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from scipy.stats import gaussian_kde
import matplotlib.font_manager as fm
from matplotlib import rcParams

# Set global font to Times New Roman
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
rcParams['mathtext.fontset'] = 'stix'

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
    

def calculate_metrics(y_true, y_pred):
    """Calculate regression metrics"""
    return {
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred)
    }

def plot_density_scatter_modern(y_true, y_pred, metrics, title, filename, subplot_label=""):
    """Draw modern style density scatter plot with density contours"""
    plt.figure(figsize=(6, 5))
    
    # Filter out points with true height of 0 but predicted height above threshold
    # These points are likely annotation errors, actually trees in reality
    threshold = 3.0  # Threshold value, if predicted height > this value but true height = 0, consider as annotation error
    valid_mask = ~((y_true == 0) & (y_pred > threshold))
    y_true_filtered = y_true[valid_mask]
    y_pred_filtered = y_pred[valid_mask]
    
    # Random sample points for visualization
    n_samples = min(100000, len(y_true_filtered))
    if n_samples < len(y_true_filtered):
        sample_idx = np.random.choice(len(y_true_filtered), n_samples, replace=False)
        y_true_sample = y_true_filtered[sample_idx]
        y_pred_sample = y_pred_filtered[sample_idx]
    else:
        y_true_sample = y_true_filtered
        y_pred_sample = y_pred_filtered

    # Calculate filtered metrics
    filtered_metrics = calculate_metrics(y_true_filtered, y_pred_filtered)

    # Create scatter plot with density
    sns.scatterplot(x=y_true_sample, y=y_pred_sample, s=3, alpha=0.6)  # s: 点的大小, edgecolor: 点边框颜色, alpha: 透明度
    
    # Add density contours
    try:
        sns.kdeplot(x=y_true_sample, y=y_pred_sample, fill=True, cmap="Spectral_r", 
                   alpha=0.03, levels=1000, thresh=0.001)  # alpha: 透明度, levels: 等高线数量, thresh: 等高线最小密度， cmap: 颜色映射,Spectral_r表示红色
    except Exception as e:
        print(f"Error in kdeplot: {e}")
        # Fallback if kdeplot fails
        pass

    # Add 1:1 line
    max_val = max(np.max(y_true_sample), np.max(y_pred_sample))
    plt.plot([0, max_val], [0, max_val], 'r--', linewidth=1.5, label="1:1 line")

    # Add regression line
    if len(y_true_filtered) > 1:
        lr = LinearRegression().fit(y_true_filtered.reshape(-1,1), y_pred_filtered)
        xlim = np.linspace(0, max_val, 100)
        plt.plot(xlim, lr.predict(xlim.reshape(-1,1)), 
                'blue', linewidth=2, label="Regression line")

    # Format text with bold subplot label if provided
    text_str = ""
    if subplot_label:
        text_str += r"$\bf{" + subplot_label + r"}$" + "\n\n"
    
    text_str += f"$R^2$: {filtered_metrics['R2']:.2f}\n"
    text_str += f"RMSE: {filtered_metrics['RMSE']:.0f}\n"
    text_str += f"MAE: {filtered_metrics['MAE']:.0f}"
    
    # Add metrics text
    plt.text(
        0.05, 0.95, text_str,
        transform=plt.gca().transAxes,
        verticalalignment='top',
        fontsize=16,
        family='Times New Roman'
    )
    
    # Set labels and font sizes
    plt.xlabel("True Height (m)", fontsize=16, family='Times New Roman')
    plt.ylabel("Predicted Height (m)", fontsize=16, family='Times New Roman')
    plt.xticks(fontsize=16, fontname='Times New Roman')
    plt.yticks(fontsize=16, fontname='Times New Roman')
    
    # Add legend
    plt.legend(loc='lower right', frameon=False, fontsize=16, prop={'family': 'Times New Roman'})
    
    # Set axis range
    plt.xlim(0, max(60, max_val))
    plt.ylim(0, max(60, max_val))
    
    plt.tight_layout()
    plt.savefig(filename, dpi=600, bbox_inches='tight')
    plt.close()
    
    # Return filtered metrics
    return filtered_metrics


# Main program
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare tree-height prediction rasters")
    parser.add_argument("true_height")
    parser.add_argument("true_mask")
    parser.add_argument("original_prediction")
    parser.add_argument("optimized_prediction")
    parser.add_argument("smoothed_prediction")
    args = parser.parse_args()

    # Set font globally
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'mathtext.fontset': 'stix'
    })
    
    # Read data
    true_height = read_tif(args.true_height).astype(float)
    true_mask = read_mask_tif(args.true_mask, [1])
    
    # Read original prediction results
    pred_height_orig = read_tif(args.original_prediction).astype(float)
    
    # Read optimized prediction results
    pred_height = read_tif(args.optimized_prediction).astype(float)
    
    # Read smoothed prediction results
    pred_height_smoothed = read_tif(args.smoothed_prediction).astype(float)

    # Unify dimensions
    shapes = [true_height.shape, pred_height.shape, true_mask.shape, 
              pred_height_orig.shape, pred_height_smoothed.shape]
    min_rows = min(s[0] for s in shapes)
    min_cols = min(s[1] for s in shapes)

    true_height = true_height[:min_rows, :min_cols]
    pred_height_orig = pred_height_orig[:min_rows, :min_cols]
    pred_height = pred_height[:min_rows, :min_cols]
    pred_height_smoothed = pred_height_smoothed[:min_rows, :min_cols]
    true_mask = true_mask[:min_rows, :min_cols]

    # Case 1: Tree areas only, original prediction
    tree_mask = true_mask == 1
    y_true_case1 = true_height[tree_mask]
    y_pred_case1 = pred_height_orig[tree_mask]
    
    # Case 2: Tree areas only, optimized prediction
    y_true_case2 = true_height[tree_mask]
    y_pred_case2 = pred_height[tree_mask]
    
    # Case 3: Tree areas only, smoothed prediction
    y_true_case3 = true_height[tree_mask]
    y_pred_case3 = pred_height_smoothed[tree_mask]
    
    # Plot and get metrics - Original Prediction
    metrics_case1 = plot_density_scatter_modern(
        y_true_case1, y_pred_case1, 
        calculate_metrics(y_true_case1, y_pred_case1),
        "Original Prediction", 
        "tree_height_original.png",
        "(a) Original"
    )
    
    # Plot and get metrics - Optimized Prediction
    metrics_case2 = plot_density_scatter_modern(
        y_true_case2, y_pred_case2, 
        calculate_metrics(y_true_case2, y_pred_case2),
        "Optimized Prediction", 
        "tree_height_optimized.png",
        "(b) Optimized"
    )
    
    # Plot and get metrics - Smoothed Prediction
    metrics_case3 = plot_density_scatter_modern(
        y_true_case3, y_pred_case3, 
        calculate_metrics(y_true_case3, y_pred_case3),
        "Smoothed Prediction", 
        "tree_height_smoothed.png",
        "(c) Smoothed"
    )

    # Print results
    print("Original Prediction Metrics:")
    print("\n".join([f"{k}: {v:.4f}" for k, v in metrics_case1.items()]))
    
    print("\nOptimized Prediction Metrics:")
    print("\n".join([f"{k}: {v:.4f}" for k, v in metrics_case2.items()]))
    
    print("\nSmoothed Prediction Metrics:")
    print("\n".join([f"{k}: {v:.4f}" for k, v in metrics_case3.items()]))
