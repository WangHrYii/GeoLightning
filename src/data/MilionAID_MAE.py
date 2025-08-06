import os
import random
import pickle
import io
from typing import Optional, Callable, List

import torch
from torch.utils.data import DataLoader, Dataset
import lightning as L
from PIL import Image
import lmdb
from tqdm import tqdm

# --- Fallback Dataset (for when LMDB is not created yet) ---
class UnifiedImageDataset(Dataset):
    """
    一个自定义的 PyTorch 数据集，从提供的文件路径列表中加载图像。
    用于处理合并和打乱后的 MillionAID 数据集。
    """
    def __init__(self, image_paths: List[str], transform: Optional[Callable] = None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"警告：无法加载图像 {img_path}。错误: {e}")
            return torch.zeros(3, 224, 224)

        if self.transform:
            image = self.transform(image)
        return image

# --- Optimized LMDB Dataset ---
class LMDBDataset(Dataset):
    """
    从预先构建的 LMDB 数据库加载图像的数据集，以优化 I/O 性能。
    """
    def __init__(self, db_path: str, keys: List[bytes], transform: Optional[Callable] = None):
        self.db_path = db_path
        self.keys = keys
        self.transform = transform
        self.env = None

    def _init_db(self):
        # 在每个 worker 进程中独立初始化 LMDB 环境
        self.env = lmdb.open(
            self.db_path,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False
        )

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx: int):
        if self.env is None:
            self._init_db()

        # 从 LMDB 中获取序列化的图像
        with self.env.begin(write=False) as txn:
            img_byte_flow = txn.get(self.keys[idx])
        
        # 使用 Pillow 从字节流中打开图像
        try:
            # 将字节流包装成文件对象
            image = Image.open(io.BytesIO(img_byte_flow)).convert('RGB')
        except Exception as e:
            # 增加鲁棒性，以防数据库中存入了损坏的图像
            print(f"警告：无法从LMDB加载索引为 {idx} 的图像。错误: {e}")
            # 返回一个占位符张量，与您的 UnifiedImageDataset 行为保持一致
            if self.transform:
                return self.transform(Image.new('RGB', (224, 224)))
            else:
                return torch.zeros(3, 224, 224)

        if self.transform:
            image = self.transform(image)
        
        return image
        
    def __del__(self):
        """确保在对象被销毁时关闭LMDB环境"""
        if self.env is not None:
            self.env.close()
            self.env = None

# --- DataModule with I/O Optimization ---
class MillionAIDDataModule(L.LightningDataModule):
    """
    用于 MillionAID 数据集的 PyTorch Lightning DataModule，专为预训练设计。
    此版本会自动检测并使用预处理的 LMDB 数据库以优化 I/O，
    如果数据库不存在，则回退到标准的文件读取模式。
    """
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 4,
        train_transform: Optional[Callable] = None,
        val_test_transform: Optional[Callable] = None,
        val_ratio: float = 0.01,
        test_ratio: float = 0.01,
        seed: int = 42,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.db_path = os.path.join(self.data_dir, "millionaid.lmdb")
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transform = train_transform
        self.val_test_transform = val_test_transform
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        self.save_hyperparameters(ignore=['train_transform', 'val_test_transform'])
        self.train_dataset, self.val_dataset, self.test_dataset = None, None, None

    def prepare_data(self):
        """检查数据目录是否存在。如果 LMDB 文件不存在，会发出警告。"""
        if not os.path.isdir(self.data_dir):
            raise FileNotFoundError(f"数据目录未找到: {self.data_dir}")
        if not os.path.exists(self.db_path):
            print("="*80)
            print(f"警告: 在 '{self.data_dir}' 中未找到 LMDB 数据库 'millionaid.lmdb'。")
            print("将使用标准的文件读取模式，这可能会非常慢。")
            print("建议运行一次预处理脚本来创建 LMDB 数据库以获得最佳性能。")
            print("您可以通过运行 `python your_datamodule_file.py --create-db` 来创建。")
            print("="*80)

    def setup(self, stage: Optional[str] = None):
        """
        优先使用 LMDB 数据库进行设置。如果不存在，则回退到标准模式。
        """
        use_lmdb = os.path.exists(self.db_path)

        if use_lmdb:
            # --- 从 LMDB 加载 ---
            env = lmdb.open(self.db_path, readonly=True, lock=False, readahead=False, meminit=False)
            try:
                with env.begin(write=False) as txn:
                    # 加载在预处理时保存的密钥列表
                    all_keys = pickle.loads(txn.get(b'__keys__'))
                
                # 为可复现的分割打乱密钥
                rng = random.Random(self.seed)
                rng.shuffle(all_keys)
            finally:
                # 确保关闭环境
                env.close()
            
            n_total = len(all_keys)
        else:
            # --- 回退到标准文件读取模式 ---
            all_image_paths = self._get_all_image_paths()
            rng = random.Random(self.seed)
            rng.shuffle(all_image_paths)
            n_total = len(all_image_paths)

        # 计算分割大小
        n_val = int(self.val_ratio * n_total)
        n_test = int(self.test_ratio * n_total)
        n_train = n_total - n_val - n_test

        if n_train <= 0 or n_val <= 0 or n_test <= 0:
            raise ValueError(f"数据集分割导致某个集合样本数为0。总数: {n_total}...")

        # 分割数据 (密钥或路径)
        if use_lmdb:
            train_keys = all_keys[:n_train]
            val_keys = all_keys[n_train : n_train + n_val]
            test_keys = all_keys[n_train + n_val :]
        else:
            train_paths = all_image_paths[:n_train]
            val_paths = all_image_paths[n_train : n_train + n_val]
            test_paths = all_image_paths[n_train + n_val :]

        # 创建相应的 Dataset 对象
        if stage == 'fit' or stage is None:
            if use_lmdb:
                self.train_dataset = LMDBDataset(self.db_path, train_keys, self.train_transform)
                self.val_dataset = LMDBDataset(self.db_path, val_keys, self.val_test_transform)
            else:
                self.train_dataset = UnifiedImageDataset(train_paths, self.train_transform)
                self.val_dataset = UnifiedImageDataset(val_paths, self.val_test_transform)
        
        if stage == 'test' or stage is None:
            if use_lmdb:
                self.test_dataset = LMDBDataset(self.db_path, test_keys, self.val_test_transform)
            else:
                self.test_dataset = UnifiedImageDataset(test_paths, self.val_test_transform)

    def _get_all_image_paths(self) -> List[str]:
        """辅助函数，用于收集所有图像的路径。"""
        all_paths = []
        train_dir = os.path.join(self.data_dir, 'train')
        test_dir = os.path.join(self.data_dir, 'test')
        for root, _, files in os.walk(train_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    all_paths.append(os.path.join(root, file))
        for file in os.listdir(test_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                all_paths.append(os.path.join(test_dir, file))
        return all_paths

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None: self.setup('fit')
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers, 
            pin_memory=True, 
            persistent_workers=True if self.num_workers > 0 else False,
            prefetch_factor=4 if self.num_workers > 0 else None
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None: self.setup('fit')
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers, 
            pin_memory=True, 
            persistent_workers=True if self.num_workers > 0 else False,
            prefetch_factor=2 if self.num_workers > 0 else None
        )

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None: self.setup('test')
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers, 
            pin_memory=True, 
            persistent_workers=True if self.num_workers > 0 else False,
            prefetch_factor=2 if self.num_workers > 0 else None
        )

def create_lmdb_database(data_dir: str):
    """
    扫描图像文件夹并创建一个 LMDB 数据库。
    """
    print("开始创建 LMDB 数据库... 这可能需要一些时间。")
    
    # 收集所有图像路径
    all_image_paths = []
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    print("正在扫描 'train' 目录...")
    for root, _, files in os.walk(train_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                all_image_paths.append(os.path.join(root, file))
    print("正在扫描 'test' 目录...")
    for file in os.listdir(test_dir):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            all_image_paths.append(os.path.join(test_dir, file))
    
    print(f"总共找到 {len(all_image_paths)} 张图像。")

    # 创建 LMDB 环境
    # map_size 需要足够大以容纳所有图像
    db_path = os.path.join(data_dir, "millionaid.lmdb")
    env = lmdb.open(db_path, map_size=549755813888)
    
    keys = []
    with env.begin(write=True) as txn:
        for i, path in enumerate(tqdm(all_image_paths, desc="正在写入数据库")):
            try:
                with open(path, 'rb') as f:
                    image_data = f.read()
                
                # 使用索引作为键
                key = f"{i:08}".encode('ascii')
                txn.put(key, image_data)
                keys.append(key)
            except Exception as e:
                print(f"\n跳过文件 {path}，错误: {e}")
        
        # 保存密钥列表以便后续加载
        txn.put(b'__keys__', pickle.dumps(keys))
        print(f"\n总共写入 {len(keys)} 张图像到数据库。")

    print(f"LMDB 数据库已成功创建于: {db_path}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="MillionAID DataModule and LMDB Creator")
    parser.add_argument(
        '--data-dir', 
        type=str, 
        required=False,
        default='/mnt/data/MillionAID',
        help="MillionAID 数据集的根目录路径。"
    )
    parser.add_argument(
        '--create-db',
        action='store_true',
        default=True,
        required=False,
        help="如果设置此标志，将创建 LMDB 数据库而不是测试数据加载器。"
    )
    args = parser.parse_args()

    if args.create_db:
    #     create_lmdb_database(args.data_dir)
    # else:
        # --- 测试数据加载器 ---
        print("测试数据加载器...")
        from torchvision import transforms
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        
        dm = MillionAIDDataModule(
            data_dir=args.data_dir,
            batch_size=8,
            num_workers=2,
            train_transform=transform,
            val_test_transform=transform
        )
        dm.prepare_data()
        dm.setup('fit')
        
        dataloader = dm.train_dataloader()
        i =0

        for batch in dataloader:
            print(batch.shape)
            i += 1
            if i > 1000:
                break