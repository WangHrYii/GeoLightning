"""
此文件使 'src.models' 目录成为一个有效的 Python 包，
便于从其他模块导入模型类。
"""

# 导入子模块，使它们可以通过 src.models 访问
from pathlib import Path
import os

def _import_submodules():
    """
    导入所有子目录作为子模块
    """
    current_dir = Path(__file__).parent
    
    # 遍历当前目录下的所有子目录
    for item in current_dir.iterdir():
        if item.is_dir() and (item / '__init__.py').exists():
            # 将目录名称作为子模块名称
            module_name = item.name
            # 将子模块导入到当前命名空间
            __import__(f"{__package__}.{module_name}")

# 执行导入
_import_submodules() 