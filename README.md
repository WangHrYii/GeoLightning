<div align="center">

<img src="images/logo_all.png" alt="GeoLightning Logo"/>

[![python](https://img.shields.io/badge/Python_3.8+|3.9|3.10-blue?logo=python)](https://www.python.org/)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch)](https://pytorch.org/)
[![lightning](https://img.shields.io/badge/Lightning_2.0+-792ee5?logo=pytorchlightning)](https://lightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)

**多任务多模态遥感深度学习框架**

</div>

## 🌐 框架概述

GeoLightning 是一个基于 PyTorch Lightning 的地理空间深度学习框架，专注于遥感图像处理与分析，融合多种数据模态与任务类型，实现高效可扩展的地球观测数据智能处理。

```yaml
meta:
  version: "2.0"
  framework_type: "multi-task multi-modal"
```

## 📂 项目结构

```yml
GeoLightning/
├── configs/                     # 配置文件目录
│   ├── configs_demo/            # 示例配置
│   ├── TreeHeightUnet_config/   # 树高估计模型配置
│   └── Unet_config/             # U-Net模型配置
├── src/                         # 核心源代码
│   ├── callbacks/               # 自定义回调
│   ├── data/                    # 数据处理模块
│   ├── losses/                  # 损失函数
│   ├── models/                  # 模型实现
│   │   ├── backbones/           # 特征提取骨干网络
│   │   │   └── bricks/          # 网络基础构建模块
│   │   ├── ChangeTask/          # 变化检测任务
│   │   ├── ClassficationTask/   # 分类任务
│   │   ├── DetectionTask/       # 目标检测任务
│   │   ├── MheadUnet/           # 多头U-Net
│   │   ├── RegresstionTask/     # 回归任务
│   │   ├── SegmentationTask/    # 分割任务
│   │   └── SelfSupervisedTask/  # 自监督任务
│   ├── preprocess/              # 数据预处理
│   ├── utils/                   # 工具函数
│   ├── eval.py                  # 评估入口
│   ├── inference.py             # 推理
│   └── train.py                 # 训练入口
├── tools/                       # 辅助工具脚本
├── tests/                       # 测试套件
└── images/                      # 示例资源
```

## 🧩 多维架构设计

GeoLightning 采用多维度架构设计，支持灵活组合与扩展：

- **任务类型维度**：支持像素级、对象级和场景级处理
- **数据模态维度**：处理光学、SAR、LiDAR等多源数据
- **处理阶段维度**：贯穿预处理、核心处理到后处理全链路
- **组件类别维度**：模块化封装可插拔组件

## 🚀 核心功能

### 骨干网络

提供多种先进的特征提取网络：

```yaml
backbone_networks:
  CNN_based:
    - "ResNet(50/101)"    # 残差网络
    - "ConvNeXt"          # 新一代卷积网络
    - "MobileNet"         # 轻量级网络

  Transformer_based:
    - "ViT"               # 视觉transformer
    - "Swin"              # 分层窗口注意力
    - "BEiT"              # 双向编码

  Specialized:
    - "UNet"              # U形编解码
    - "HRNet"             # 高分辨率网络
    - "BiSeNet"           # 双向分割网络
```

### 任务支持

集成多种遥感任务处理能力：

```yaml
task_support:
  Segmentation:           # 分割任务
    - "语义分割"           # 像素级分类
    - "实例分割"           # 对象区分

  Detection:              # 检测任务
    - "目标检测"           # 目标定位与分类
    - "变化检测"           # 多时相差异识别

  Regression:             # 回归任务
    - "树高估计"           # 植被高度推断
    - "生物量计算"         # 生物质量预测

  Classification:         # 分类任务
    - "地物分类"           # 土地覆盖分类
    - "场景识别"           # 场景类型判断
```

### 数据处理能力

专为地理空间数据设计的处理管线：

```yaml
data_processing:
  spatial:
    - "地理编码保持"       # 保留地理参考
    - "坐标转换"          # 不同坐标系处理
    - "多分辨率融合"       # 尺度适配

  tiling:
    - "智能分块"          # 自适应分块策略
    - "无缝拼接"          # 边缘融合
    - "大图推理"          # 超大影像处理

  augmentation:
    - "地理特定增强"       # 针对遥感特性
    - "多尺度训练"        # 尺度不变性增强
    - "谱间变换"          # 波段操作
```

## ⚡ 高级特性

### 分块推理引擎

处理超大遥感影像的关键技术：

- **自适应分块**：根据模型与显存动态调整块大小
- **位置敏感拼接**：边缘优化的无缝拼接技术
- **批处理调度**：内存友好的推理排程
- **地理参考保持**：全流程保持空间参考一致性

### 多模态融合策略

支持多源遥感数据的融合方法：

- **早期融合**：输入层数据融合
- **特征融合**：中间层特征交互
- **结果融合**：决策级集成
- **跨模态注意力**：不同模态间的自适应加权

### 集成Lightning优势

充分利用PyTorch Lightning生态：

- **分布式训练**：多GPU/多节点无缝扩展
- **混合精度**：自动FP16训练加速
- **实验追踪**：集成主流实验管理工具
- **检查点管理**：智能模型保存与恢复
- **进度可视化**：训练进程实时监控

## ⏭️ 待实现功能 (TODO)

### 近期计划

| ✅ 已完成           | ⬜ 待完成           |
|---------------------|--------------------|
| ✅ 模块化骨干网络实现 | ⬜ 轻量级骨干网络优化 |
| ✅ 分块推理基础功能   | ⬜ 边缘融合算法优化   |
| ✅ 语义分割任务支持   | ⬜ 实例分割任务扩展   |

### 中期计划

| ⬜ 计划功能         | ⬜ 关联功能         |
|---------------------|--------------------|
| ⬜ 多模态数据融合框架 | ⬜ 异构数据统一表示   |
| ⬜ 时间序列分析支持   | ⬜ 变化轨迹追踪能力   |
| ⬜ 模型量化与剪枝     | ⬜ ONNX/TensorRT导出  |

### 远期愿景

| ⬜ 基础设施         | ⬜ 核心功能         |
|---------------------|--------------------|
| ⬜ 云原生部署架构     | ⬜ 分布式处理框架     |
| ⬜ 条件影像生成模型   | ⬜ 样本合成与增强     |
| ⬜ GIS软件插件开发    | ⬜ API服务化接口      |

## 💼 应用领域

- **城市规划**：建筑检测、土地利用分析
- **农业监测**：作物分类、生长监测、产量预测
- **环境保护**：森林覆盖变化、水体监测
- **灾害评估**：洪水范围、火灾影响、地质灾害
- **基础设施管理**：道路网络提取、变电站检测

## 🔧 安装与使用

```bash
# 安装依赖
pip install -r requirements.txt

# 训练模型
python src/train.py

# 评估模型
python src/eval.py
```

## 🤝 贡献

欢迎提交Pull Requests或Issues改进框架。详情请参阅[贡献指南](CONTRIBUTING.md)。

## 📄 许可证

[MIT License](LICENSE)
