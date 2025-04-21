import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch_optimizer import RAdam

class Registry:
    def __init__(self):
        self._dict = {}

    def register(self, name=None):
        def decorator(obj):
            key = name or obj.__name__
            if key in self._dict:
                raise ValueError(f"{key} is already registered.")
            self._dict[key] = obj
            return obj
        return decorator

    def get(self, name):
        return self._dict.get(name)

MODEL_REGISTRY = Registry()
NETWORK_REGISTRY = Registry()
LOSS_REGISTRY = Registry()
OPTIMIZER_REGISTRY = Registry()
SCHEDULER_REGISTRY = Registry()
AUGMENTATION_REGISTRY = Registry()
DATASET_REGISTRY = Registry()


# # 数据增强注册器
# AUGMENTATION_REGISTRY = {
#     "basic": transforms.Compose([
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomVerticalFlip(),
#         transforms.RandomRotation(30),
#         transforms.ToTensor(),
#     ]),
#     "advanced": transforms.Compose([
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomCrop(224),
#         transforms.ColorJitter(brightness=0.5, contrast=0.5),
#         transforms.ToTensor(),
#     ]),
#     # 其他数据增强方法
# }

OPTIMIZER_REGISTRY = {
    "sgd": lambda params, config: torch.optim.SGD(
        params,
        lr=config.get('lr', 0.01),
        momentum=config.get('momentum', 0.9),
        weight_decay=config.get('weight_decay', 0.0)
    ),
    "adam": lambda params, config: torch.optim.Adam(
        params,
        lr=config.get('lr', 0.001),
        weight_decay=config.get('weight_decay', 0.0)
    ),
    "radam": lambda params, config: RAdam(
        params,
        lr=config.get('lr', 0.001),
        weight_decay=config.get('weight_decay', 0.0)
    ),
    "adamw": lambda params, config: torch.optim.AdamW(
        params,
        lr=config.get('lr', 0.001),
        weight_decay=config.get('weight_decay', 0.0)
    )
}

SCHEDULER_REGISTRY = {
    "freezed": lambda optimizer, config: torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epochs: config.get('max_lr', 0.001)),
    "step_lr": lambda optimizer, config: torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.get('step_size', 30), gamma=config.get('gamma', 0.1)),
    "multistep_lr": lambda optimizer, config: torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.get('steps', [30, 80]), gamma=config.get('gamma', 0.1)),
    "exp_lr": lambda optimizer, config: torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.get('gamma', 0.9)),
    "cos_lr": lambda optimizer, config: torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.get('cos_lr_t', 100), eta_min=config.get('gamma', 1e-6)),
    "cos_restart": lambda optimizer, config: torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=config.get('cos_lr_t', 10), T_mult=config.get('cos_t_mult', 2), eta_min=1e-6),
    "one_cycle": lambda optimizer, config: torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config.get('max_lr', 0.01), total_steps=config.get('total_steps', 1), epochs=config.get('total_epoches', 100), anneal_strategy="cos", final_div_factor=config.get('gamma', 1e4)),
    "plateau_lr": lambda optimizer, config: torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=config.get('gamma', 0.1), patience=config.get('patience', 10)),
    "poly_lr": lambda optimizer, config: torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0 - epoch * (1 - config.get('gamma', 0.1)) / config.get('total_epoches', 100))
}