import os
from typing import Optional, Callable

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import ImageFolder
import lightning as L  # <--- 修改: 从 'pytorch_lightning as pl' 改为 'lightning as L'
from PIL import Image

# --- Custom Dataset Classes ---

class UnlabeledImageFolder(ImageFolder):
    """
    A variant of ImageFolder that returns only the image, discarding the label.
    This is useful for self-supervised pre-training tasks like MAE where
    labels are not required.
    """
    def __getitem__(self, index: int) -> torch.Tensor:
        """
        Overrides the default __getitem__ to return only the image.
        
        Args:
            index (int): The index of the sample.
            
        Returns:
            torch.Tensor: The transformed image tensor.
        """
        # The original __getitem__ returns (sample, target)
        sample, _ = super().__getitem__(index)
        return sample

class UnlabeledImageDataset(Dataset):
    """
    Used to load an unlabeled image folder, such as the MillionAID test set.
    """
    def __init__(self, data_dir: str, transform: Optional[Callable] = None):
        """
        Args:
            data_dir (str): Directory path containing the images.
            transform (callable, optional): Optional transform to be applied to a sample.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(data_dir) if self._is_image_file(f)])

    def _is_image_file(self, filename: str) -> bool:
        """Checks if a file is a supported image type."""
        return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))

    def __len__(self) -> int:
        """Returns the total number of samples."""
        return len(self.image_files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Retrieves a sample. Since there are no labels, it returns only the image.
        
        Args:
            idx (int): The sample index.
            
        Returns:
            torch.Tensor: The transformed image.
        """
        img_path = os.path.join(self.data_dir, self.image_files[idx])
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Warning: Could not load image {img_path}. Error: {e}")
            # Return a placeholder tensor if an image is corrupt
            return torch.zeros(3, 224, 224) # Assuming input size is 224x224

        if self.transform:
            image = self.transform(image)
            
        return image

# --- PyTorch Lightning DataModule ---
class MillionAIDDataModule(L.LightningDataModule): # <--- 修改: 继承自 L.LightningDataModule
    """
    PyTorch Lightning DataModule for the MillionAID dataset, tailored for
    self-supervised pre-training.
    
    This DataModule automatically handles loading for training, validation, and test sets.
    It splits the original 'train' folder and ensures all dataloaders return only
    image tensors, which is ideal for models like MAE.
    """
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 4,
        train_transform: Optional[Callable] = None,
        val_test_transform: Optional[Callable] = None,
        val_split: float = 0.1,
        seed: int = 42,
    ):
        """
        Args:
            data_dir (str): Root directory of the MillionAID dataset.
                           It should contain 'train' and 'test' subdirectories.
            batch_size (int): Number of samples per batch.
            num_workers (int): Number of subprocesses to use for data loading.
            train_transform (callable, optional): Image transform for the training set.
            val_test_transform (callable, optional): Image transform for validation and test sets.
            val_split (float): The fraction of the training data to use for validation.
            seed (int): Random seed for reproducible splits.
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transform = train_transform
        self.val_test_transform = val_test_transform
        self.val_split = val_split
        self.seed = seed

        self.save_hyperparameters(ignore=['train_transform', 'val_test_transform'])

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        """
        Actions to be performed on a single process, like checking for data.
        """
        train_path = os.path.join(self.data_dir, 'train')
        test_path = os.path.join(self.data_dir, 'test')
        if not os.path.exists(train_path) or not os.path.exists(test_path):
            raise FileNotFoundError(
                f"Could not find 'train' or 'test' folders in '{self.data_dir}'. "
                "Please ensure the MillionAID dataset is downloaded and extracted correctly."
            )

    def setup(self, stage: Optional[str] = None):
        """
        Assigns train/val/test datasets for each GPU process.
        
        Args:
            stage (str, optional): One of 'fit', 'validate', 'test', 'predict'.
                                   Determines which datasets to set up.
        """
        if stage == 'fit' or stage is None:
            # 1. Use UnlabeledImageFolder to load the full training dataset without labels
            full_dataset = UnlabeledImageFolder(
                os.path.join(self.data_dir, 'train'),
                transform=None # Transforms are applied after the split
            )
            
            # 2. Calculate split sizes
            n_samples = len(full_dataset)
            n_val = int(self.val_split * n_samples)
            n_train = n_samples - n_val
            
            # 3. Split the dataset
            self.train_dataset, self.val_dataset = random_split(
                full_dataset, [n_train, n_val],
                generator=torch.Generator().manual_seed(self.seed)
            )
            
            # 4. Apply the respective transforms to the underlying dataset of each subset
            self.train_dataset.dataset.transform = self.train_transform
            self.val_dataset.dataset.transform = self.val_test_transform

        if stage == 'test' or stage is None:
            # 5. Set up the test dataset
            self.test_dataset = UnlabeledImageDataset(
                os.path.join(self.data_dir, 'test'),
                transform=self.val_test_transform
            )
            
        if stage == 'validate':
             # Set up validation set if only validating
            if self.val_dataset is None:
                full_dataset = UnlabeledImageFolder(os.path.join(self.data_dir, 'train'), transform=None)
                n_samples = len(full_dataset)
                n_val = int(self.val_split * n_samples)
                n_train = n_samples - n_val
                _, self.val_dataset = random_split(full_dataset, [n_train, n_val], generator=torch.Generator().manual_seed(self.seed))
                self.val_dataset.dataset.transform = self.val_test_transform

    def train_dataloader(self) -> DataLoader:
        """Creates the training dataloader."""
        if self.train_dataset is None:
            raise RuntimeError("Train dataset is not set up. Call setup('fit') first.")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self) -> DataLoader:
        """Creates the validation dataloader."""
        if self.val_dataset is None:
            raise RuntimeError("Validation dataset is not set up. Call setup('fit') or setup('validate') first.")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def test_dataloader(self) -> DataLoader:
        """Creates the test dataloader."""
        if self.test_dataset is None:
            raise RuntimeError("Test dataset is not set up. Call setup('test') first.")
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )