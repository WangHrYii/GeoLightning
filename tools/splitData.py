import numpy as np
from osgeo import gdal
import geopandas as gpd
import os
from pathlib import Path

def get_tree_heights(nDSM_data, tree_mask):
    """
    Get tree heights where tree_mask is 1
    """
    # Get tree heights by applying the mask
    tree_heights = np.zeros(nDSM_data.shape)
    tree_heights[tree_mask == 1] = nDSM_data[tree_mask == 1]
    return tree_heights

def read_raster(file_path):
    """
    Read raster file using GDAL and return the data as numpy array
    """
    dataset = gdal.Open(file_path)
    if dataset is None:
        raise Exception(f"Could not open {file_path}")
    
    band = dataset.GetRasterBand(1)
    data = band.ReadAsArray()
    return data, dataset.GetGeoTransform()

def process_entire_area(nDSM_path, tree_cover_path, output_path):
    """
    Process the entire area and save tree heights
    """
    # Read the data using GDAL
    nDSM_data, _ = read_raster(nDSM_path)
    tree_mask, _ = read_raster(tree_cover_path)
    
    # Get tree heights
    tree_heights = get_tree_heights(nDSM_data, tree_mask)
    
    # Save to npy file
    np.save(output_path, tree_heights)
    print(f"Saved entire area tree heights to {output_path}")

def process_grid_cells(nDSM_path, tree_cover_path, grid_path, output_dir):
    """
    Process each grid cell and save individual tree heights
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the grid shapefile
    grid = gpd.read_file(grid_path)
    
    # Open raster datasets
    nDSM_dataset = gdal.Open(nDSM_path)
    tree_dataset = gdal.Open(tree_cover_path)
    
    if nDSM_dataset is None or tree_dataset is None:
        raise Exception("Could not open raster files")
    
    # Process each grid cell
    for idx, row in grid.iterrows():
        # Get the grid cell geometry
        geom = row.geometry
        
        # Get the bounding box of the geometry
        minx, miny, maxx, maxy = geom.bounds
        
        # Convert bounds to pixel coordinates
        nDSM_transform = nDSM_dataset.GetGeoTransform()
        x_origin = nDSM_transform[0]
        y_origin = nDSM_transform[3]
        pixel_width = nDSM_transform[1]
        pixel_height = nDSM_transform[5]  # 这是负值
        
        # 计算像素坐标，注意y坐标的计算
        x_start = int((minx - x_origin) / pixel_width)
        x_end = int((maxx - x_origin) / pixel_width)
        
        # 由于pixel_height是负值，所以miny对应y_end，maxy对应y_start
        y_start = int((maxy - y_origin) / pixel_height)  # 使用maxy
        y_end = int((miny - y_origin) / pixel_height)    # 使用miny
        
        # 确保坐标顺序正确
        if x_start > x_end:
            x_start, x_end = x_end, x_start
        if y_start > y_end:
            y_start, y_end = y_end, y_start
        
        # Ensure coordinates are within bounds
        x_start = max(0, x_start)
        y_start = max(0, y_start)
        x_end = min(nDSM_dataset.RasterXSize, x_end)
        y_end = min(nDSM_dataset.RasterYSize, y_end)
        
        # Read the data for this region
        nDSM_band = nDSM_dataset.GetRasterBand(1)
        tree_band = tree_dataset.GetRasterBand(1)
        
        nDSM_data = nDSM_band.ReadAsArray(x_start, y_start, 
                                         x_end - x_start, y_end - y_start)
        tree_mask = tree_band.ReadAsArray(x_start, y_start, 
                                        x_end - x_start, y_end - y_start)
        
        try:
            # Get tree heights for this grid cell
            tree_heights = get_tree_heights(nDSM_data, tree_mask)
            
            # Save to npy file using OBJECTID_1
            output_path = os.path.join(output_dir, f"grid_{row['OBJECTID_1']}.npy")
            np.save(output_path, tree_heights)
            print(f"Processed grid {row['OBJECTID_1']}")
            
        except Exception as e:
            print(f"Error processing grid {row['OBJECTID_1']}: {str(e)}")
    
    # Close datasets
    nDSM_dataset = None
    tree_dataset = None

def main():
    # Define input and output paths
    nDSM_path = "/mnt/data/TreeHeight/nDSM_cropped_boundary.tif"
    tree_cover_path = "/mnt/data/TreeHeight/treecover/treecover_reproj_cropped.tif"
    grid_path = "/mnt/data/TreeHeight/dc_reprojected.shp"
    output_dir = "/mnt/data/TreeHeight/tree_height_data"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Process entire area
    process_entire_area(nDSM_path, tree_cover_path, 
                       os.path.join(output_dir, "entire_area_heights.npy"))
    
    # Process grid cells
    process_grid_cells(nDSM_path, tree_cover_path, grid_path, output_dir)

if __name__ == "__main__":
    main()
