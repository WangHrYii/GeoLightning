# from .ce_loss import *
# from .dice_loss import *
# from .focal_loss import *
# from .lovasz_loss import *

# 如果网络文件较多，可以使用自动导入
import os
import glob
from pathlib import Path

def _import_networks():
    current_dir = Path(__file__).parent
    
    # 递归搜索所有 .py 文件
    def import_from_dir(directory):
        for item in directory.glob('**/*.py'):
            if item.is_file() and not item.name.endswith(("__init__.py")):
                # 构建相对于包的导入路径
                rel_path = item.relative_to(current_dir)
                # 将路径分隔符替换为点号
                module_path = str(rel_path).replace(os.sep, '.')[:-3]  # 移除 .py
                # 导入模块
                __import__(f"{__package__}.{module_path}")
    
    import_from_dir(current_dir)

_import_networks()