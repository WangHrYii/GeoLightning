# data_patcher.py (version 2)
import os
from osgeo import gdal
from tqdm import tqdm
import math

gdal.UseExceptions()

def _generate_patches_for_single_split(split_input_dir, split_output_dir, patch_size, overlap_ratio):
    """
    (Internal helper function) Generates patches for a single data split 
    (e.g., 'train') with a specific overlap.
    """
    stride = math.ceil(patch_size * (1 - overlap_ratio))
    print(f"  Patch Size: {patch_size}x{patch_size}, Overlap: {overlap_ratio*100:.0f}%, Stride: {stride} pixels")

    for type_folder in os.listdir(split_input_dir):
        type_path = os.path.join(split_input_dir, type_folder)
        if not os.path.isdir(type_path):
            continue

        output_patch_dir = os.path.join(split_output_dir, type_folder)
        os.makedirs(output_patch_dir, exist_ok=True)
        
        grid_files = [f for f in os.listdir(type_path) if f.endswith(('.tif', '.tiff'))]

        for grid_filename in tqdm(grid_files, desc=f"  Patching {type_folder}"):
            grid_path = os.path.join(type_path, grid_filename)
            try:
                ds = gdal.Open(grid_path)
                if ds is None: continue
                width, height = ds.RasterXSize, ds.RasterYSize
                
                patch_row_idx = 0
                for y in range(0, height, stride):
                    if y + patch_size > height: y = height - patch_size
                    patch_col_idx = 0
                    for x in range(0, width, stride):
                        if x + patch_size > width: x = width - patch_size
                        
                        base_name = os.path.splitext(grid_filename)[0]
                        patch_filename = f"{base_name}_p_{patch_row_idx}_{patch_col_idx}.tif"
                        output_path = os.path.join(output_patch_dir, patch_filename)
                        
                        gdal.Translate(output_path, ds, srcWin=[x, y, patch_size, patch_size])
                        
                        patch_col_idx += 1
                        if x + patch_size >= width: break
                    patch_row_idx += 1
                    if y + patch_size >= height: break
                ds = None
            except Exception as e:
                print(f"\nError processing file {grid_path}: {e}")

def generate_patches_with_split_awareness(input_dir, output_dir, patch_size=256, train_overlap=0.25):
    """
    Cuts grid images into patches with different strategies for train, val, and test sets.
    - Training set uses overlap for data augmentation.
    - Validation and Test sets use zero overlap for fair evaluation.

    Args:
        input_dir (str): Root directory of clipped grids ('clipped_data/').
        output_dir (str): Root directory to save the new patches ('patched_data/').
        patch_size (int): The edge size of the patches.
        train_overlap (float): The overlap ratio for the training set only.
    """
    print("--- Starting Patch Generation with Split-Aware Strategy ---")
    
    splits_config = {
        'train': train_overlap,
        'val': 0.0,
        'test': 0.0
    }

    for split_name, overlap in splits_config.items():
        split_input_path = os.path.join(input_dir, split_name)
        split_output_path = os.path.join(output_dir, split_name)

        if not os.path.exists(split_input_path):
            print(f"Warning: Directory for split '{split_name}' not found at {split_input_path}. Skipping.")
            continue

        print(f"\nProcessing split: '{split_name}'")
        _generate_patches_for_single_split(
            split_input_dir=split_input_path,
            split_output_dir=split_output_path,
            patch_size=patch_size,
            overlap_ratio=overlap
        )

    print("\n--- All patching tasks completed successfully! ---")


# ===================================================================
# ===================      HOW TO USE       =========================
# ===================================================================
if __name__ == '__main__':
    # 1. 配置路径和参数
    INPUT_CLIPPED_DIR = "/mnt/data/TreeHeight/DistributionSplitData/"
    OUTPUT_PATCHES_DIR = "/mnt/data/TreeHeight/DistributionSplitData_Patched/"

    PATCH_SIZE = 256
    # 只需为训练集指定重叠率，验证和测试集将自动使用0重叠
    TRAIN_OVERLAP_RATIO = 0.25

    # 2. 调用主函数
    if "path/to/your" in INPUT_CLIPPED_DIR:
        print("="*60)
        print("!! 请先在脚本的 `if __name__ == '__main__':` 部分修改文件路径 !!")
        print("="*60)
    else:
        generate_patches_with_split_awareness(
            input_dir=INPUT_CLIPPED_DIR,
            output_dir=OUTPUT_PATCHES_DIR,
            patch_size=PATCH_SIZE,
            train_overlap=TRAIN_OVERLAP_RATIO
        )
