import os
import argparse
from pathlib import Path

def generate_tree(directory, prefix='', is_last=True, exclude_dirs=None, output_file=None):
    """
    生成目录树结构
    :param directory: 目标目录路径
    :param prefix: 前缀字符串（用于缩进）
    :param is_last: 是否为最后一个子项
    :param exclude_dirs: 需要排除的目录列表
    :param output_file: 输出文件对象
    """
    exclude_dirs = exclude_dirs or []
    directory = Path(directory)
    
    # 获取目录下所有条目并排序
    entries = sorted([entry for entry in directory.iterdir()], key=lambda x: (not x.is_dir(), x.name.lower()))
    
    # 过滤排除目录
    entries = [entry for entry in entries if entry.name not in exclude_dirs and not entry.name.startswith('.')]
    
    # 确定连接符号
    connector = '└── ' if is_last else '├── '
    
    # 打印当前目录名称
    line = f"{prefix}{connector}{directory.name}/"
    _output(line, output_file)
    
    # 更新前缀
    new_prefix = prefix + ('    ' if is_last else '│   ')
    
    for index, entry in enumerate(entries):
        is_last_entry = index == len(entries) - 1
        
        if entry.is_dir():
            generate_tree(
                entry, 
                new_prefix, 
                is_last_entry, 
                exclude_dirs, 
                output_file
            )
        else:
            _output(f"{new_prefix}{'└── ' if is_last_entry else '├── '}{entry.name}", output_file)

def _output(line, output_file):
    """统一输出方法"""
    print(line)
    if output_file:
        output_file.write(line + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='生成目录树结构')
    parser.add_argument('root_dir', nargs='?', default='.', help='根目录路径（默认为当前目录）')
    parser.add_argument('-o', '--output', help='输出到文件')
    parser.add_argument('-e', '--exclude', nargs='+', default=[], help='排除的目录列表')
    
    args = parser.parse_args()
    
    output_file = None
    if args.output:
        output_file = open(args.output, 'w', encoding='utf-8')
    
    try:
        print(f"生成目录树: {args.root_dir}")
        generate_tree(
            args.root_dir,
            exclude_dirs=args.exclude,
            output_file=output_file
        )
    except Exception as e:
        print(f"错误: {str(e)}")
    finally:
        if output_file:
            output_file.close()
