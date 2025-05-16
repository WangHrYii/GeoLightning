from pytorch_lightning import Callback
from osgeo import gdal
import numpy as np

### PredictionCallback Class
class RSListedPredCallback(Callback):
    """
    A custom PyTorch Lightning Callback for handling predictions on remote sensing images.

    This callback is designed to process predictions made on large remote sensing images
    by dividing them into smaller patches, predicting on each patch, and then stitching
    the predictions back together into a single output image.

    Attributes:
        img_path (str): Path to the input image file.
        result_path (str): Path where the output prediction image will be saved.
        patch_size (int): Size of the patches (in pixels) that the image is divided into.
        stride (int): Stride (in pixels) used when sliding the window to extract patches.
        predictions (list): List to store predictions for each patch.
        patch_positions (list): List to store the positions (x, y) of each patch in the original image.
    """

    def __init__(self, img_path, result_path, patch_size=512, stride=256):
        """
        Initializes the RSListedPredCallback with the given parameters.

        Args:
            img_path (str): Path to the input image file.
            result_path (str): Path where the output prediction image will be saved.
            patch_size (int, optional): Size of the patches (in pixels). Defaults to 512.
            stride (int, optional): Stride (in pixels) used when sliding the window. Defaults to 256.
        """
        super().__init__()
        self.stride      = stride
        self.img_path    = img_path
        self.result_path = result_path
        self.patch_size  = patch_size
        self.predictions = []
        self.patch_positions = []

    def on_predict_start(self, trainer, pl_module):
        """
        Called when the prediction process starts.

        Resets the predictions and patch_positions lists to prepare for a new prediction run.

        Args:
            trainer (pl.Trainer): The PyTorch Lightning trainer instance.
            pl_module (pl.LightningModule): The PyTorch Lightning module being used for prediction.
        """
        self.predictions = []
        self.patch_positions = trainer.datamodule.dataset.patch_positions

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        """
        Called at the end of each prediction batch.

        Stores the predictions for each batch in the predictions list.

        Args:
            trainer (pl.Trainer): The PyTorch Lightning trainer instance.
            pl_module (pl.LightningModule): The PyTorch Lightning module being used for prediction.
            outputs (torch.Tensor): The output predictions from the model.
            batch (Any): The batch of data that was predicted on.
            batch_idx (int): The index of the current batch.
            dataloader_idx (int): The index of the current dataloader.
        """
        preds = outputs.detach().cpu().numpy()
        self.predictions.extend(preds)

    def on_predict_end(self, trainer, pl_module):
        """
        Called when the prediction process ends.

        Stitches the predictions from all patches back into a single image and saves the result.

        Args:
            trainer (pl.Trainer): The PyTorch Lightning trainer instance.
            pl_module (pl.LightningModule): The PyTorch Lightning module being used for prediction.
        """
        dataset = gdal.Open(self.img_path)
        assert dataset is not None, f"Could not open {self.img_path}"
        geotransform = dataset.GetGeoTransform()
        projection   = dataset.GetProjection()
        height       = dataset.RasterYSize
        width        = dataset.RasterXSize

        result = np.zeros((height, width), dtype=np.float32)
        count = np.zeros((height, width), dtype=np.float32)

        for pred, (x, y) in zip(self.predictions, self.patch_positions):
            pred = pred.squeeze()
            pred = np.clip(pred, 0, 1)  # Ensure values are between 0 and 1
            pred = (pred * 255).astype(np.uint8)
            result[y:y+self.patch_size, x:x+self.patch_size] += pred
            count[y:y+self.patch_size, x:x+self.patch_size] += 1

        result[count > 0] = result[count > 0] / count[count > 0]
        result = result.astype(np.uint8)

        driver = gdal.GetDriverByName('GTiff')
        out_dataset = driver.Create(self.result_path, width, height, 1, gdal.GDT_Byte)
        out_dataset.SetGeoTransform(geotransform)
        out_dataset.SetProjection(projection)
        out_dataset.GetRasterBand(1).WriteArray(result)
        out_dataset.FlushCache()
        out_dataset = None 