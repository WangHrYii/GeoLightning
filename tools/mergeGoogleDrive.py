import os
import shutil
import argparse
from pathlib import Path

def merge_leaf_folders(source_dir, target_dir):
    """
    合并只包含文件的叶子文件夹到目标目录
    
    参数:
        source_dir: 源目录路径
        target_dir: 目标目录路径
    """
    # 确保目标目录存在
    os.makedirs(target_dir, exist_ok=True)
    
    leaf_folders_count = 0
    files_count = 0
    
    # 遍历源目录
    for root, dirs, files in os.walk(source_dir):
        # 如果当前目录没有子目录但有文件，则认为它是叶子文件夹
        if not dirs and files:
            leaf_folders_count += 1
            folder_name = os.path.basename(root)
            # 创建以文件夹名命名的目标子目录
            dest_folder = os.path.join(target_dir, folder_name)
            os.makedirs(dest_folder, exist_ok=True)
            
            print(f"合并文件夹: {root} -> {dest_folder}")
            
            # 复制所有文件到目标目录
            for file in files:
                files_count += 1
                src_file_path = os.path.join(root, file)
                dest_file_path = os.path.join(dest_folder, file)
                
                # 如果目标文件已存在，添加数字后缀
                if os.path.exists(dest_file_path):
                    base, ext = os.path.splitext(file)
                    counter = 1
                    while os.path.exists(os.path.join(dest_folder, f"{base}_{counter}{ext}")):
                        counter += 1
                    dest_file_path = os.path.join(dest_folder, f"{base}_{counter}{ext}")
                
                try:
                    # 复制文件
                    shutil.copy2(src_file_path, dest_file_path)
                    print(f"  复制文件: {file}")
                except (shutil.Error, IOError) as e:
                    print(f"  复制文件 {file} 时出错: {e}")
    
    return leaf_folders_count, files_count

def get_input_directory(prompt, default=None):
    """获取用户输入的目录路径并验证"""
    while True:
        if default:
            user_input = input(f"{prompt} [默认: {default}]: ").strip() or default
        else:
            user_input = input(f"{prompt}: ").strip()
        
        if not user_input:
            print("请输入有效的目录路径")
            continue
            
        # 展开用户主目录和环境变量
        expanded_path = os.path.expanduser(os.path.expandvars(user_input))
        
        if os.path.isdir(expanded_path):
            return expanded_path
        else:
            create_dir = input(f"目录 '{expanded_path}' 不存在，是否创建? (y/n): ").lower()
            if create_dir == 'y':
                try:
                    os.makedirs(expanded_path, exist_ok=True)
                    return expanded_path
                except OSError as e:
                    print(f"无法创建目录: {e}")
            else:
                print("请提供一个有效的目录路径")

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='合并只包含文件的文件夹到指定目标目录')
    parser.add_argument('--source_dir', help='源目录路径')
    parser.add_argument('--target_dir', help='目标目录路径（默认为源目录下的merged_folders）')
    
    args = parser.parse_args()
    
    # 如果命令行未提供源目录，交互式获取
    source_dir = args.source_dir
    if not source_dir:
        source_dir = get_input_directory("请输入源目录路径")
    
    # 如果命令行未提供目标目录，使用默认值或交互式获取
    target_dir = args.target_dir
    if not target_dir:
        default_target = os.path.join(source_dir, 'merged_folders')
        use_default = input(f"是否使用默认目标目录 '{default_target}'? (y/n): ").lower()
        if use_default == 'y':
            target_dir = default_target
        else:
            target_dir = get_input_directory("请输入目标目录路径")
    
    # 最终确认
    print(f"\n准备合并操作:")
    print(f"源目录: {source_dir}")
    print(f"目标目录: {target_dir}")
    confirm = input("是否继续? (y/n): ").lower()
    if confirm != 'y':
        print("操作已取消")
        return
    
    try:
        leaf_folders_count, files_count = merge_leaf_folders(source_dir, target_dir)
        print(f"\n合并完成!")
        print(f"处理了 {leaf_folders_count} 个叶子文件夹")
        print(f"复制了 {files_count} 个文件")
        print(f"所有内容已合并到 {target_dir}")
    except Exception as e:
        print(f"合并过程中发生错误: {e}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n操作被用户中断")
    except Exception as e:
        print(f"发生未预期的错误: {e}")