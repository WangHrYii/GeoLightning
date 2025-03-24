import os
import zipfile
import shutil
from pathlib import Path

def merge_google_drive(zip_parent_dir, target_folders, output_parent_dir):
    """
    Unzips and merges Google Drive zip files into target folders
    
    Args:
        zip_parent_dir (str): Path to directory containing zip files
        target_folders (list): List of folder names to merge into
        output_parent_dir (str): Parent directory for merged folders
    """
    # Convert to Path objects
    zip_parent_dir = Path(zip_parent_dir)
    output_parent_dir = Path(output_parent_dir)
    
    # Create output parent directory if it doesn't exist
    output_parent_dir.mkdir(parents=True, exist_ok=True)

    # Create target folders under output parent directory
    for folder in target_folders:
        target_dir = output_parent_dir / folder   # 可以直接用 / 来组合路径，因为 Path 类重载了 / 运算符
        target_dir.mkdir(exist_ok=True)

    # Process each zip file
    for zip_file in zip_parent_dir.glob('*.zip'):
        print(f'Processing {zip_file.name}...')

        # Create temp directory for extraction
        temp_dir = Path('temp_extract')
        temp_dir.mkdir(exist_ok=True)
        
        try:
            # Extract zip file
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)

            # Merge contents into target folders
            for root, dirs, files in os.walk(temp_dir):
                for dir_name in dirs:
                    if dir_name in target_folders:
                        src = Path(root) / dir_name
                        dst = output_parent_dir / dir_name
                        
                        # Copy contents
                        for item in src.glob('*'):
                            if item.is_file():
                                shutil.copy2(item, dst)
                            elif item.is_dir():
                                shutil.copytree(item, dst / item.name)
            
        finally:
            # Clean up temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)


def merge_identical_folders(zip_parent_dir, output_dir):
    """
    Merges zip files containing identical folder structures
    
    Args:
        zip_parent_dir (str): Path to directory containing zip files
        output_dir (str): Directory to merge all folders into
    """
    zip_dir = Path(zip_parent_dir)
    output_dir = Path(output_dir)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    for zip_file in zip_dir.glob('*.zip'):
        print(f'Processing {zip_file.name}...')
        
        temp_dir = Path('temp_extract')
        temp_dir.mkdir(exist_ok=True)
        
        try:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # Walk through extracted files
            for root, dirs, files in os.walk(temp_dir):
                for dir_name in dirs:
                    src_dir = Path(root) / dir_name
                    dst_dir = output_dir / dir_name
                    
                    # Create target directory if needed
                    dst_dir.mkdir(exist_ok=True)
                    
                    # Copy contents
                    for item in src_dir.glob('*'):
                        if item.is_file():
                            shutil.copy2(item, dst_dir)
                        elif item.is_dir():
                            shutil.copytree(item, dst_dir / item.name)
                            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


def checkPathFiles(file_path):
    # 生成本来应该有的文件路径
    id = range(0, 391)
    pre_text = 'Sentinel2_'
    post_text = ['Q1', 'Q2', 'Q3', 'Q4']
    not_found = []
    for i in id:
        for j in post_text:
            file_name = pre_text + str(i) + '_' + j + '.tif'
            file_path = Path(file_path)
            file = file_path / file_name
            if not file.exists():
                print(file_name, 'not found')
                not_found.append(file_name)
    return not_found


if __name__ == '__main__':
    # zip_parent_dir = '/mnt/data_B/Industry_image_train/raw_zips'
    # target_folders = ['GEE_Export', 'GEE_Export_4Q']
    # output_parent_dir = '/mnt/data_B/Industry_image_train/merged'

    # merge_google_drive(zip_parent_dir, target_folders, output_parent_dir)
    # checkPathFiles('/mnt/data_B/Industry_image_train/merged/GEE_Export_4Q')

    zip_parent_dir = '/mnt/data_B/Industry_image_infra/'
    output_dir = '/mnt/data_B/Industry_image_infra_merged'
    merge_identical_folders(zip_parent_dir, output_dir)