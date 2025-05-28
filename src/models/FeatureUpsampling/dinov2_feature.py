from lightning import LightningModule
from src.models.backbones.dinov2 import DINOv2
import torch


class DINOv2FeatureExtractor(LightningModule):
    def __init__(self, model_name, checkpoint_path):
        super().__init__()
        self.model = DINOv2(model_name)
        self.model.load_state_dict(torch.load(checkpoint_path))
        self.model.eval()


    def forward(self, x):
        with torch.no_grad():
            return self.model.get_intermediate_layers(x, reshape=True)[0]
