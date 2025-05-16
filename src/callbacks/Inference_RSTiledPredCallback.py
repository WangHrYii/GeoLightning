import torch
import numpy as np
from osgeo import gdal
from torch.utils.data import Dataset, DataLoader
import lightning as pl
from typing import Optional, List, Tuple, Any

class PredictionCallback(pl.Callback):
    def __init__(
        self,
        output_path: str = "output.tif",
        overlap: float = 0.25,
        task_names: List[str] = None
    ):
        super().__init__()
        self.output_path = output_path
        self.overlap = overlap
        self.task_names = task_names
        self.pred_buffer = []
        self.coord_buffer = []

    def on_predict_start(self, trainer, pl_module):
        self.pred_buffer = []
        self.coord_buffer = []

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        preds, coords = outputs
        
        # 处理多任务输出，添加通道维度
        if isinstance(preds, (tuple, list)):
            task_preds = []
            for p in preds:
                p_np = p.cpu().numpy()
                if p_np.ndim == 3:  # [batch, h, w] -> [batch, 1, h, w]
                    p_np = np.expand_dims(p_np, axis=1)
                task_preds.append(p_np)
        else:
            p_np = preds.cpu().numpy()
            if p_np.ndim == 3:
                p_np = np.expand_dims(p_np, axis=1)
            task_preds = [p_np]
            
        self.pred_buffer.append(task_preds)
        self.coord_buffer.extend(coords)

    def on_predict_epoch_end(self, trainer, pl_module):
        assert len(self.pred_buffer) != 0, "没有预测结果"

        # 重组数据为 [task][batch]
        all_task_preds = list(zip(*self.pred_buffer))
        num_tasks = len(all_task_preds)
        
        # 验证任务名称配置
        if self.task_names is None:
            self.task_names = [f"task_{i}" for i in range(num_tasks)]
        assert len(self.task_names) == num_tasks, "任务名称数量与输出不匹配"

        # 获取数据集信息
        dataset = trainer.datamodule.dataset
        orig_shape = dataset.original_img.shape
        pad_top, pad_left = dataset.padding
        patch_size = dataset.patch_size
        window = self._create_window(patch_size)

        # 为每个任务重建图像
        for task_idx, task_preds in enumerate(all_task_preds):
            # 合并该任务的所有批次预测
            merged_preds = np.concatenate(task_preds, axis=0)
            
            # 初始化重建缓冲区
            c = merged_preds.shape[1]  # 当前任务的通道数
            padded_shape = (c,) + dataset.padded_img.shape[1:]
            output = np.zeros(padded_shape, dtype=np.float32)
            counter = np.zeros(padded_shape[1:], dtype=np.float32)  # HxW

            # 加权合并每个预测块
            for pred, (y, x) in zip(merged_preds, self.coord_buffer):
                weighted_pred = pred * window  # [C, H, W] * [1, H, W]
                output[:, y:y+patch_size, x:x+patch_size] += weighted_pred
                counter[y:y+patch_size, x:x+patch_size] += window[0]

            # 归一化处理
            counter[counter < 1e-8] = 1.0
            output = output / counter[np.newaxis, ...]

            # 裁剪填充区域
            final_output = output[
                :,
                pad_top:pad_top+orig_shape[1],
                pad_left:pad_left+orig_shape[2]
            ]

            # 保存任务结果
            self._save_geotiff(
                array=final_output.transpose(1, 2, 0),
                geotrans=dataset.geotrans,
                proj=dataset.proj,
                task_name=self.task_names[task_idx]
            )

    def _create_window(self, size: int) -> np.ndarray:
        window = np.hanning(size)[:, None] * np.hanning(size)[None, :]
        return window[np.newaxis, ...]  # 保持通道维度

    def _save_geotiff(self, array: np.ndarray, geotrans, proj, task_name: str):
        filename = self.output_path.replace(".tif", f"_{task_name}.tif")
        driver = gdal.GetDriverByName('GTiff')
        h, w, c = array.shape
        
        dataset = driver.Create(
            filename,
            w,
            h,
            c,
            gdal.GDT_Float32
        )
        dataset.SetGeoTransform(geotrans)
        dataset.SetProjection(proj)
        
        for i in range(c):
            dataset.GetRasterBand(i+1).WriteArray(array[..., i])
        
        dataset.FlushCache() 