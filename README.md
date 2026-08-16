<div align="center">

<img src="images/logo_all.png" alt="GeoLightning Logo"/>

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.13.0-ee4c2c?logo=pytorch)](https://pytorch.org/)
[![Lightning](https://img.shields.io/badge/Lightning-fcef404-792ee5?logo=lightning)](https://lightning.ai/)
[![Hydra](https://img.shields.io/badge/Hydra-1.3.2-89b8cd)](https://hydra.cc/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**面向遥感影像的多任务、多模态 PyTorch Lightning 框架**

</div>

GeoLightning 提供遥感分类、语义分割、树高回归、自监督学习、特征上采样和
SAR-光学多模态任务所需的数据、模型、训练与大图推理组件。当前版本为
`0.2.0`，主要运行栈固定为 PyTorch `2.13.0`、TorchVision `0.28.0` 和
包含上游 checkpoint 安全修复的 Lightning 提交 `fcef4045`。

## 核心能力

| 组件 | 当前能力 |
| --- | --- |
| 数据集 | 源码集成 TorchGeo `v0.4.1` 的 85 个数据集类，不依赖外部 `torchgeo` 包 |
| 空间采样 | Random、RandomBatch、Grid 和 PreChipped GeoSampler |
| Backbone | 仓库原有遥感骨干网络，加上 11 个 TorchVision 源码家族、52 个模型变体 |
| 统一接口 | `forward_features()`、`out_channels`、`out_strides` 特征协议 |
| 任务 | 分类、分割、树高回归、多头任务、SAR-光学融合、AutoMAE、特征上采样 |
| 工程能力 | Hydra 配置、Lightning 训练、分块推理、CI、wheel 构建和输出保留策略 |

## 项目结构

```text
GeoLightning/
├── configs/
│   ├── AutoMAE/                  # AutoMAE 预训练
│   ├── TreeHeight_DPT/           # MINTHE/DPT 树高任务
│   ├── TreeHeight_Unet/          # 多头 U-Net 树高任务
│   ├── RSIPAC_25_T1/             # SAR-光学分割
│   ├── torchgeo/                 # TorchGeo 数据集与 DataModule 示例
│   └── backbones/                # Backbone 示例
├── src/
│   ├── data/
│   │   └── torchgeo/             # 本地 TorchGeo datasets 与 samplers 源码
│   ├── models/
│   │   ├── backbones/            # 原有与源码集成 backbone
│   │   ├── MultiHeadTask/        # 多头分割/高度回归
│   │   ├── MultiModalSegTask/    # SAR-光学多模态分割
│   │   ├── SegmentationTask/     # 语义分割
│   │   ├── RegressionTask/       # 回归组件
│   │   ├── SelfSupervisedTask/   # AutoMAE 等自监督模型
│   │   └── FeatureUpsampling/    # 特征上采样
│   ├── train.py                  # 默认训练入口
│   ├── eval.py                   # 评估入口
│   └── inference.py              # 推理入口
├── requirements/                 # 分层依赖与版本约束
├── tests/                        # CPU 测试基线
├── tools/                        # 数据工具和输出清理工具
└── docs/                         # 集成与存储策略说明
```

## 安装

支持 Python `3.10`。建议先安装与机器 CUDA 版本匹配的 PyTorch，
再安装 GeoLightning 功能依赖。

CPU 环境：

```bash
python -m pip install torch==2.13.0 torchvision==0.28.0 \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[train,geo]"
```

完整开发与全部可选数据读取器：

```bash
python -m pip install -e ".[train,geo,datasets,dev]"
```

也可以使用分层 requirements：

```bash
python -m pip install -r requirements/train.txt
python -m pip install -r requirements/geo.txt
```

根目录的 `requirements.txt` 会安装完整开发环境。各依赖层说明见
[requirements/README.md](requirements/README.md)。

## 路径配置

仓库配置不包含开发机器绝对路径。数据、权重和项目目录通过环境变量或 Hydra
override 指定，常用变量记录在 [.env.example](.env.example) 中：

```dotenv
PROJECT_ROOT=.
DATA_ROOT=./data
CHECKPOINT_ROOT=./ckpts
TREEHEIGHT_PATCH_ROOT=./data/TreeHeight/DistributionSplitData_Patched
DEPTH_WEIGHTS=./ckpts/depth_anything_v2_vitb.pth
```

`rootutils` 会读取项目根目录的 `.env`。所有 Hydra 参数也可以在命令行覆盖。

## 训练、评估与推理

安装为 editable package 后可以直接使用 CLI：

```bash
# 默认使用 configs/TreeHeight_DPT/config.yaml
geolightning-train

# 从检查点评估
geolightning-eval ckpt_path=/path/to/model.ckpt

# 分块预测
geolightning-inference ckpt_path=/path/to/model.ckpt
```

常用 Hydra override 示例：

```bash
geolightning-train \
  train_data.batch_size=8 \
  trainer.accelerator=cpu \
  trainer.devices=1
```

其他任务入口：

```bash
python src/train_AutoMAE.py
python src/train_rsipac_mcanet.py
python src/inference_tiled.py
```

## TorchGeo 源码数据集

TorchGeo `v0.4.1` 的 dataset 与 sampler 实现位于
`src/data/torchgeo/`。数据集类按需加载，导入目录时不会一次性加载所有地理和
可选读取依赖。

```python
from src.data import torchgeo

print(len(torchgeo.available_datasets()))  # 85

EuroSAT = torchgeo.get_dataset_class("EuroSAT")
dataset = EuroSAT(
    root="data/eurosat",
    split="train",
    bands=("B04", "B03", "B02"),
)
```

索引型数据集可以通过统一 Lightning DataModule 使用：

```yaml
_target_: src.data.TorchGeoDataModule
dataset_name: EuroSAT
root: ${oc.env:DATA_ROOT,data}/eurosat
batch_size: 64
common_kwargs:
  bands: [B04, B03, B02]
  download: false
train_kwargs: {split: train}
val_kwargs: {split: val}
test_kwargs: {split: test}
```

空间型 `GeoDataset` 必须配置 sampler。训练可使用随机批采样，验证和测试可使用
确定性网格采样：

```yaml
train_sampler:
  kind: random_batch
  size: 256
  length: 4096
  units: pixels
val_sampler:
  kind: grid
  size: 256
  stride: 256
  units: pixels
```

完整配置见
[EuroSAT DataModule](configs/torchgeo/eurosat_datamodule.yaml) 和
[Chesapeake13 空间采样](configs/torchgeo/chesapeake13_datamodule.yaml)。

核心地理数据集需要 Rasterio、Fiona、PyProj、Shapely 和 Rtree。部分数据集还
需要 `h5py`、`laspy`、`pycocotools` 等可选依赖。依赖 Radiant MLHub 下载的
少数数据集需在独立环境安装 `.[mlhub-legacy]`；该停止维护的客户端固定依赖
Pydantic 1 和 Shapely 1.8，不能与标准 `train,geo` 环境混装。

## Backbone 源码集成

`src/models/backbones/torchvision_source/` 包含 TorchVision `v0.28.0` 的本地
模型定义、构造器、注册表与权重元数据，没有调用 `torchvision.models`。
TorchVision 安装包仅提供底层算子、变换和权重下载工具。

源码家族包括 AlexNet、DenseNet、EfficientNet、GoogLeNet、Inception、
MaxVit、MNASNet、RegNet、ShuffleNetV2、SqueezeNet 和 VGG。

```python
import torch

from src.models.backbones.torchvision_source import TorchvisionSourceBackbone

encoder = TorchvisionSourceBackbone(
    model_name="efficientnet_b0",
    in_channels=6,
    pretrained=False,
)

features = encoder(torch.randn(2, 6, 256, 256))
print(encoder.out_channels)
print(encoder.out_strides)
```

仓库原有 backbone 可以通过统一适配器转换为同一特征协议：

```python
from src.models.backbones import adapt_backbone

encoder = adapt_backbone(existing_encoder)
features = encoder.forward_features(images)
```

更多细节见 [docs/source-integrations.md](docs/source-integrations.md)。

## Checkpoint 与输出保留

维护中的训练配置默认只保留一个监控指标最优 checkpoint 和一个 `last.ckpt`。
历史 Hydra run 使用以下命令预览清理计划：

```bash
python tools/prune_outputs.py outputs --keep-latest 20 --max-age-days 90
```

工具默认不会删除文件。确认预览结果后显式添加 `--apply`；重要 run 可以在目录
内创建 `.keep` 文件保护。完整规则见
[docs/storage-policy.md](docs/storage-policy.md)。

## 测试与构建

```bash
pytest
python -m build
python -m pip check
python tools/export_audit_requirements.py /tmp/geolightning-audit.txt
pip-audit --disable-pip --no-deps -r /tmp/geolightning-audit.txt
pip-audit --skip-editable --ignore-vuln PYSEC-2026-3624
```

GitHub Actions 会在 CPU 环境安装锁定依赖、构建 wheel、运行非慢速测试，并分别
审计直接依赖版本与完整安装环境。`PYSEC-2026-3624` 的忽略仅用于处理版本元数据
误报：Lightning 固定提交 `fcef4045` 已包含对应上游修复 `d710d689`，测试套件
也会验证不受信任的 checkpoint instantiator 被实际拦截。
当前测试覆盖数据目录惰性加载、普通与空间 DataModule、backbone 特征协议、
源码模型前向、bricks 兼容层、配置可移植性和输出保留策略。

## 许可证与第三方源码

GeoLightning 使用 [MIT License](LICENSE)。源码集成部分继续遵循各自上游许可：

- TorchGeo datasets 和 samplers：MIT
- TorchVision model sources：BSD 3-Clause

版本、commit 和本地修改范围记录在
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
