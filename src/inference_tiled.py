#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAR-Optical语义分割推理脚本（处理已分割的切片）
支持从两个文件夹中读取SAR和光学切片进行推理
"""

from typing import Any, Dict, List, Tuple
import hydra
import rootutils
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils import (
    RankedLogger,
    extras,
    instantiate_loggers,
    instantiate_callbacks,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def inference_tiled(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    对已分割的切片进行语义分割推理

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Tuple[dict, dict] with metrics and dict with all instantiated objects.
    """
    assert cfg.ckpt_path

    log.info(f"Instantiating datamodule <{cfg.inference_data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.inference_data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger, callbacks=callbacks)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    log.info("Starting tiled inference!")
    
    trainer.predict(model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)

    log.info("Tiled inference completed!")
    
    return {}, object_dict


@hydra.main(version_base="1.3", config_path="../configs/RSIPAC_25_T1/MCANet", config_name="inference_tiled.yaml")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for tiled inference.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    inference_tiled(cfg)


if __name__ == "__main__":
    main() 