
import os
from osgeo import gdal, ogr
from tqdm import tqdm

# 推荐在脚本开始时调用，让GDAL在出错时返回None而不是抛出Python异常
# 这样我们可以更好地控制错误处理
gdal.UseExceptions()

def _create_output_dirs(base_dir, splits, subfolders):
    """(Internal) Creates all necessary output directories."""
    print("Creating output directories...")
    for split in splits:
        for subfolder in subfolders:
            path = os.path.join(base_dir, split, subfolder)
            os.makedirs(path, exist_ok=True)
    print("Directories created.")

def _clip_rasters_for_split(split_name, shp_path, source_rasters, output_dir):
    """(Internal) Clips all source rasters based on a given shapefile."""
    print(f"\nProcessing split: {split_name}")
    
    if not os.path.exists(shp_path):
        print(f"Warning: Shapefile not found for split '{split_name}': {shp_path}. Skipping.")
        return

    dataSource = ogr.Open(shp_path, 0)
    if dataSource is None:
        print(f"Error: Could not open {shp_path}")
        return
        
    layer = dataSource.GetLayer()
    feature_count = layer.GetFeatureCount()
    print(f"Found {feature_count} grids in {os.path.basename(shp_path)}.")

    for i in tqdm(range(feature_count), desc=f"Clipping for {split_name}"):
        feature = layer.GetFeature(i)
        fid = feature.GetFID()

        for raster_type, raster_path in source_rasters.items():
            if not os.path.exists(raster_path):
                print(f"Warning: Source raster not found: {raster_path}. Skipping this raster type.")
                continue

            out_folder = os.path.join(output_dir, split_name, raster_type)
            out_path = os.path.join(out_folder, f"grid_{fid}.tif")

            try:
                gdal.Warp(
                    out_path,
                    raster_path,
                    cutlineDSName=shp_path,
                    cutlineWhere=f"FID = {fid}",
                    cropToCutline=True,
                    dstNodata=0 # 你可以根据需要修改或移除此项
                )
            except Exception as e:
                print(f"\nError clipping grid FID {fid} for {raster_type}: {e}")
                print(f"  Input Raster: {raster_path}")
                print(f"  Shapefile: {shp_path}")
                print("Skipping this grid.")

    dataSource = None
    print(f"Finished processing for {split_name}.")

def process_and_clip_data(rgb_path, ndsm_path, mask_path, shp_dir, output_dir):
    """
    Main function to clip large rasters based on grid shapefiles.

    This function reads train/val/test shapefiles, uses the polygons within them
    to clip source RGB, nDSM, and TreeMask rasters, and saves the results
    into a structured output directory.

    Args:
        rgb_path (str): Path to the source RGB mosaic TIF file.
        ndsm_path (str): Path to the source nDSM mosaic TIF file.
        mask_path (str): Path to the source TreeMask mosaic TIF file.
        shp_dir (str): Directory containing train.shp, val.shp, and test.shp.
        output_dir (str): Root directory to save the clipped images.
    """
    print("--- Starting Raster Clipping Process ---")
    
    # --- 1. Configuration ---
    source_rasters = {
        "RGB": rgb_path,
        "nDSM": ndsm_path,
        "TreeMask": mask_path
    }
    splits = ["train", "val", "test"]
    subfolders = ["RGB", "nDSM", "TreeMask"]

    # --- 2. Create Output Directories ---
    _create_output_dirs(output_dir, splits, subfolders)

    # --- 3. Process each split (train, val, test) ---
    for split in splits:
        shp_file_path = os.path.join(shp_dir, f"{split}_data.shp")
        _clip_rasters_for_split(split, shp_file_path, source_rasters, output_dir)

    print("\n--- All clipping tasks completed successfully! ---")


# ===================================================================
# ===================      HOW TO USE       =========================
# ===================================================================
if __name__ == '__main__':
    # 1. 在这里配置你的文件路径
    # !! 修改成你的实际路径 !!
    SOURCE_RGB_PATH = "/mnt/data/TreeHeight/raster_1m_cropped.tif"
    SOURCE_NDSM_PATH = "/mnt/data/TreeHeight/nDSM_cropped_boundary.tif"
    SOURCE_MASK_PATH = "/mnt/data/TreeHeight/treecover/treecover_reproj_cropped.tif"
    SHAPEFILES_DIRECTORY = "/mnt/data/TreeHeight/SplitData_SHP/"
    OUTPUT_DIRECTORY = "/mnt/data/TreeHeight/DistributionSplitData/"

    # 2. 调用主函数执行所有操作
    # 检查路径是否已修改
    if "path/to/your" in SOURCE_RGB_PATH:
        print("="*60)
        print("!! 请先在脚本的 `if __name__ == '__main__':` 部分修改文件路径 !!")
        print("="*60)
    else:
        process_and_clip_data(
            rgb_path=SOURCE_RGB_PATH,
            ndsm_path=SOURCE_NDSM_PATH,
            mask_path=SOURCE_MASK_PATH,
            shp_dir=SHAPEFILES_DIRECTORY,
            output_dir=OUTPUT_DIRECTORY
        )

