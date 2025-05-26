import os
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from osgeo import gdal
import random
from scipy.stats import wasserstein_distance
from tqdm import tqdm
from os.path import join, isfile, isdir
import shutil


def read_tif(file_path):
    """Read a GeoTIFF file and return its data as a numpy array"""
    dataset = gdal.Open(file_path)
    if dataset is None:
        raise ValueError(f"Cannot open {file_path}")
    band = dataset.GetRasterBand(1)
    data = band.ReadAsArray()
    return data, dataset


def calculate_tree_height_distribution(ndsm_path, tree_cover_path, num_bins=50, mask_shapefile=None):
    """
    Calculate the height distribution histogram of trees in the nDSM file
    
    Args:
        ndsm_path: Path to nDSM.tif (height data)
        tree_cover_path: Path to treeCover.tif (mask of tree areas)
        num_bins: Number of bins for the histogram
        mask_shapefile: Optional shapefile path to clip the rasters
        
    Returns:
        heights: Array of tree heights
        hist: Histogram values
        bin_edges: Edges of histogram bins
    """
    # If mask shapefile is provided, clip the rasters first
    if mask_shapefile:
        # Create temporary files for clipped rasters
        temp_ndsm = "temp_ndsm.tif"
        temp_tree_cover = "temp_tree_cover.tif"
        
        # Clip rasters with shapefile
        print(f"Clipping nDSM with shapefile: {mask_shapefile}")
        gdal.Warp(temp_ndsm, ndsm_path, cutlineDSName=mask_shapefile, cropToCutline=True, dstNodata=0)
        
        print(f"Clipping tree cover with shapefile: {mask_shapefile}")
        gdal.Warp(temp_tree_cover, tree_cover_path, cutlineDSName=mask_shapefile, cropToCutline=True, dstNodata=0)
        
        # Use clipped rasters
        ndsm_data, _ = read_tif(temp_ndsm)
        tree_mask, _ = read_tif(temp_tree_cover)
        
        # Clean up temporary files
        os.remove(temp_ndsm)
        os.remove(temp_tree_cover)
    else:
        print(f"Reading nDSM from {ndsm_path}")
        ndsm_data, _ = read_tif(ndsm_path)
        
        print(f"Reading tree cover mask from {tree_cover_path}")
        tree_mask, _ = read_tif(tree_cover_path)
    
    # Apply mask to get only tree heights
    tree_heights = ndsm_data[tree_mask == 1]
    
    # Remove invalid heights (e.g., negative values or extremely large values)
    valid_heights = tree_heights[(tree_heights > 0) & (tree_heights < 100)]
    
    # If no valid heights found (e.g., no trees in this region), return empty arrays
    if len(valid_heights) == 0:
        return np.array([]), np.array([]), np.linspace(0, 50, num_bins+1)
    
    # Calculate histogram
    hist, bin_edges = np.histogram(valid_heights, bins=num_bins, density=True)
    
    return valid_heights, hist, bin_edges


def plot_histogram(heights, hist, bin_edges, output_path=None):
    """Plot and optionally save the height distribution histogram"""
    plt.figure(figsize=(10, 6))
    plt.hist(heights, bins=bin_edges, density=True, alpha=0.7, color='green')
    plt.xlabel('Tree Height (m)')
    plt.ylabel('Density')
    plt.title('Tree Height Distribution')
    plt.grid(alpha=0.3)
    
    if output_path:
        plt.savefig(output_path)
        print(f"Histogram saved to {output_path}")
    plt.close()


def split_grids(shp_path, output_dir, ndsm_path, tree_cover_path, 
                train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, 
                max_attempts=10, tolerance=0.05):
    """
    Split the grid shapefile into train, validation, and test sets
    while preserving the original tree height distribution.
    
    Args:
        shp_path: Path to the grid shapefile
        output_dir: Directory to save the split shapefiles
        ndsm_path: Path to nDSM.tif
        tree_cover_path: Path to treeCover.tif
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        max_attempts: Maximum number of random split attempts
        tolerance: Maximum allowed Wasserstein distance between distributions
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the grid shapefile
    print(f"Reading grid shapefile from {shp_path}")
    grid_gdf = gpd.read_file(shp_path)
    
    # Calculate global tree height distribution
    heights, global_hist, bin_edges = calculate_tree_height_distribution(ndsm_path, tree_cover_path)
    plot_histogram(heights, global_hist, bin_edges, join(output_dir, "global_height_distribution.png"))
    
    # Get total number of grids
    n_grids = len(grid_gdf)
    n_train = int(n_grids * train_ratio)
    n_val = int(n_grids * val_ratio)
    n_test = n_grids - n_train - n_val
    
    print(f"Total grids: {n_grids}, Train: {n_train}, Val: {n_val}, Test: {n_test}")
    
    best_split = None
    best_distance = float('inf')
    
    for attempt in tqdm(range(max_attempts), desc="Finding optimal split"):
        # Randomly shuffle indices
        indices = list(range(n_grids))
        random.shuffle(indices)
        
        # Split indices
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train+n_val]
        test_indices = indices[n_train+n_val:]
        
        # Create GeoDataFrames for each split
        train_gdf = grid_gdf.iloc[train_indices].copy()
        val_gdf = grid_gdf.iloc[val_indices].copy()
        test_gdf = grid_gdf.iloc[test_indices].copy()
        
        # Check distribution similarity
        # We'll clip the nDSM and tree cover by each split
        # and calculate the height distribution for each
        
        # This is a simplified approach - in practice, 
        # you would clip nDSM and tree cover by each split
        # and calculate the height distribution
        # For simplicity, we'll just use the grid sizes as a proxy
        
        # Calculate a proxy for how well the distribution is preserved
        # (Using grid sizes as a rough proxy for area coverage)
        train_size = len(train_gdf)
        val_size = len(val_gdf)
        test_size = len(test_gdf)
        
        # This is just an approximate way to evaluate distribution similarity
        # In a real implementation, you would clip the actual data
        dist1 = abs(train_size/n_grids - train_ratio)
        dist2 = abs(val_size/n_grids - val_ratio) 
        dist3 = abs(test_size/n_grids - test_ratio)
        
        total_dist = dist1 + dist2 + dist3
        
        if total_dist < best_distance:
            best_distance = total_dist
            best_split = (train_indices, val_indices, test_indices)
        
        # Check if good enough
        if total_dist < tolerance:
            print(f"Found acceptable split at attempt {attempt+1}")
            break
    
    # Use the best split found
    train_indices, val_indices, test_indices = best_split
    train_gdf = grid_gdf.iloc[train_indices].copy()
    val_gdf = grid_gdf.iloc[val_indices].copy()
    test_gdf = grid_gdf.iloc[test_indices].copy()
    
    # Save the splits
    train_path = join(output_dir, "train.shp")
    val_path = join(output_dir, "val.shp")
    test_path = join(output_dir, "test.shp")
    
    train_gdf.to_file(train_path)
    val_gdf.to_file(val_path)
    test_gdf.to_file(test_path)
    
    print(f"Train set saved to {train_path} ({len(train_gdf)} grids)")
    print(f"Validation set saved to {val_path} ({len(val_gdf)} grids)")
    print(f"Test set saved to {test_path} ({len(test_gdf)} grids)")
    
    return train_path, val_path, test_path


def advanced_split_grids(shp_path, output_dir, ndsm_path, tree_cover_path, 
                         train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, 
                         max_attempts=20, num_bins=50):
    """
    Advanced method to split the grid shapefile while preserving distribution.
    This method will clip the nDSM and tree cover by each potential split
    and calculate the actual tree height distribution.
    
    Args:
        shp_path: Path to the grid shapefile
        output_dir: Directory to save the split shapefiles
        ndsm_path: Path to nDSM.tif
        tree_cover_path: Path to treeCover.tif
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        max_attempts: Maximum number of random split attempts
        num_bins: Number of bins for histograms
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the grid shapefile
    print(f"Reading grid shapefile from {shp_path}")
    grid_gdf = gpd.read_file(shp_path)
    
    # Calculate global tree height distribution
    heights, global_hist, bin_edges = calculate_tree_height_distribution(ndsm_path, tree_cover_path, num_bins=num_bins)
    plot_histogram(heights, global_hist, bin_edges, join(output_dir, "global_height_distribution.png"))
    
    # Get total number of grids
    n_grids = len(grid_gdf)
    n_train = int(n_grids * train_ratio)
    n_val = int(n_grids * val_ratio)
    n_test = n_grids - n_train - n_val
    
    print(f"Total grids: {n_grids}, Train: {n_train}, Val: {n_val}, Test: {n_test}")
    
    # Create temporary directory for intermediate files
    temp_dir = join(output_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    best_split = None
    best_distance = float('inf')
    
    for attempt in tqdm(range(max_attempts), desc="Finding optimal split"):
        # Randomly shuffle indices
        indices = list(range(n_grids))
        random.shuffle(indices)
        
        # Split indices
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train+n_val]
        test_indices = indices[n_train+n_val:]
        
        # Create GeoDataFrames for each split
        train_gdf = grid_gdf.iloc[train_indices].copy()
        val_gdf = grid_gdf.iloc[val_indices].copy()
        test_gdf = grid_gdf.iloc[test_indices].copy()
        
        # Save temporary shapefiles
        train_temp_path = join(temp_dir, f"train_temp_{attempt}.shp")
        val_temp_path = join(temp_dir, f"val_temp_{attempt}.shp")
        test_temp_path = join(temp_dir, f"test_temp_{attempt}.shp")
        
        train_gdf.to_file(train_temp_path)
        val_gdf.to_file(val_temp_path)
        test_gdf.to_file(test_temp_path)
        
        # Calculate tree height distributions for each split
        _, train_hist, _ = calculate_tree_height_distribution(
            ndsm_path, tree_cover_path, num_bins=num_bins, mask_shapefile=train_temp_path)
        
        _, val_hist, _ = calculate_tree_height_distribution(
            ndsm_path, tree_cover_path, num_bins=num_bins, mask_shapefile=val_temp_path)
        
        _, test_hist, _ = calculate_tree_height_distribution(
            ndsm_path, tree_cover_path, num_bins=num_bins, mask_shapefile=test_temp_path)
        
        # Handle empty histograms (no trees in split)
        if len(train_hist) == 0 or len(val_hist) == 0 or len(test_hist) == 0:
            continue
            
        # Calculate Earth Mover's Distance (Wasserstein) between distributions
        train_dist = wasserstein_distance(global_hist, train_hist)
        val_dist = wasserstein_distance(global_hist, val_hist)
        test_dist = wasserstein_distance(global_hist, test_hist)
        
        # Weight distances by dataset proportions
        total_dist = train_ratio * train_dist + val_ratio * val_dist + test_ratio * test_dist
        
        print(f"Attempt {attempt+1}: EMD = {total_dist:.4f} (Train: {train_dist:.4f}, Val: {val_dist:.4f}, Test: {test_dist:.4f})")
        
        if total_dist < best_distance:
            best_distance = total_dist
            best_split = (train_indices, val_indices, test_indices)
            
            # Save histograms for best split so far
            plt.figure(figsize=(12, 8))
            plt.subplot(2, 2, 1)
            plt.bar(range(len(global_hist)), global_hist, alpha=0.7, color='blue')
            plt.title('Global Distribution')
            
            plt.subplot(2, 2, 2)
            plt.bar(range(len(train_hist)), train_hist, alpha=0.7, color='green')
            plt.title(f'Train Distribution (EMD: {train_dist:.4f})')
            
            plt.subplot(2, 2, 3)
            plt.bar(range(len(val_hist)), val_hist, alpha=0.7, color='orange')
            plt.title(f'Validation Distribution (EMD: {val_dist:.4f})')
            
            plt.subplot(2, 2, 4)
            plt.bar(range(len(test_hist)), test_hist, alpha=0.7, color='red')
            plt.title(f'Test Distribution (EMD: {test_dist:.4f})')
            
            plt.tight_layout()
            plt.savefig(join(output_dir, f"distribution_comparison_attempt_{attempt+1}.png"))
            plt.close()
    
    # Use the best split found
    train_indices, val_indices, test_indices = best_split
    train_gdf = grid_gdf.iloc[train_indices].copy()
    val_gdf = grid_gdf.iloc[val_indices].copy()
    test_gdf = grid_gdf.iloc[test_indices].copy()
    
    # Save the final splits
    train_path = join(output_dir, "train.shp")
    val_path = join(output_dir, "val.shp")
    test_path = join(output_dir, "test.shp")
    
    train_gdf.to_file(train_path)
    val_gdf.to_file(val_path)
    test_gdf.to_file(test_path)
    
    # Calculate and save final distributions
    _, train_hist, _ = calculate_tree_height_distribution(
        ndsm_path, tree_cover_path, num_bins=num_bins, mask_shapefile=train_path)
    
    _, val_hist, _ = calculate_tree_height_distribution(
        ndsm_path, tree_cover_path, num_bins=num_bins, mask_shapefile=val_path)
    
    _, test_hist, _ = calculate_tree_height_distribution(
        ndsm_path, tree_cover_path, num_bins=num_bins, mask_shapefile=test_path)
    
    # Calculate final EMD scores
    train_dist = wasserstein_distance(global_hist, train_hist)
    val_dist = wasserstein_distance(global_hist, val_hist)
    test_dist = wasserstein_distance(global_hist, test_hist)
    total_dist = train_ratio * train_dist + val_ratio * val_dist + test_ratio * test_dist
    
    # Plot final distributions comparison
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    plt.bar(range(len(global_hist)), global_hist, alpha=0.7, color='blue')
    plt.title('Global Distribution')
    
    plt.subplot(2, 2, 2)
    plt.bar(range(len(train_hist)), train_hist, alpha=0.7, color='green')
    plt.title(f'Train Distribution (EMD: {train_dist:.4f})')
    
    plt.subplot(2, 2, 3)
    plt.bar(range(len(val_hist)), val_hist, alpha=0.7, color='orange')
    plt.title(f'Validation Distribution (EMD: {val_dist:.4f})')
    
    plt.subplot(2, 2, 4)
    plt.bar(range(len(test_hist)), test_hist, alpha=0.7, color='red')
    plt.title(f'Test Distribution (EMD: {test_dist:.4f})')
    
    plt.suptitle(f'Final Split Distributions (Total EMD: {total_dist:.4f})')
    plt.tight_layout()
    plt.savefig(join(output_dir, "final_distribution_comparison.png"))
    plt.close()
    
    # Clean up temporary directory
    shutil.rmtree(temp_dir)
    
    print(f"Train set saved to {train_path} ({len(train_gdf)} grids)")
    print(f"Validation set saved to {val_path} ({len(val_gdf)} grids)")
    print(f"Test set saved to {test_path} ({len(test_gdf)} grids)")
    print(f"Final distribution similarity score (EMD): {total_dist:.4f}")
    print(f"Individual EMD scores - Train: {train_dist:.4f}, Val: {val_dist:.4f}, Test: {test_dist:.4f}")
    
    return train_path, val_path, test_path


if __name__ == "__main__":
    # Example usage
    ndsm_path = input("Enter path to nDSM.tif: ")
    tree_cover_path = input("Enter path to treeCover.tif: ")
    grid_shp_path = input("Enter path to dc.shp: ")
    output_dir = input("Enter output directory: ")
    
    # Calculate and plot the global tree height distribution
    heights, hist, bin_edges = calculate_tree_height_distribution(ndsm_path, tree_cover_path)
    plot_histogram(heights, hist, bin_edges, join(output_dir, "global_height_distribution.png"))
    
    # Uncomment one of the following methods:
    
    # Simple method (faster but less accurate)
    # print("\nUsing simplified method to split grids:")
    # split_grids(grid_shp_path, output_dir, ndsm_path, tree_cover_path)
    
    # Advanced method with actual distribution calculation (recommended)
    print("\nUsing advanced method to split grids:")
    advanced_split_grids(grid_shp_path, output_dir, ndsm_path, tree_cover_path)
