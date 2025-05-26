import os
from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import rootutils
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
import torch.optim.lr_scheduler


rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)  # 作用是将项目根目录添加到PYTHONPATH中，这样就可以直接import项目中的模块了
from src.utils import (
    RankedLogger,              # 用于记录日志
    extras,                    # 用于处理额外的配置
    get_metric_value,          # 用于获取优化的指标值
    instantiate_callbacks,     # 用于实例化回调
    instantiate_loggers,       # 用于实例化日志器
    log_hyperparameters,       # 用于记录超参数
    task_wrapper,              # 用于装饰任务，经过装饰的任务
)
# from src.models.FeatureUpsampling.LoftUp import LoftUpStage1Trainer

log = RankedLogger(__name__, rank_zero_only=True) # rank_zero_only=True 表示只有rank为0的进程才会记录日志


@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.train_data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.train_data)   # 通过hydra实例化数据模块，为什么可以这样实例化呢？因为在配置文件中已经指定了数据模块的类名

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger, precision=cfg.trainer.precision)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule)  # 开始训练, 真正的训练入口

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path  # 获取最佳模型的路径
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs/LoftUp_config", config_name="segmentation_test.yaml")  # 通过hydra.main装饰器指定配置文件的路径和名称，返回配置后的字典
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)   # 用于处理额外的配置，比如打印配置树，如果没有提供标签，则询问标签等

    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
