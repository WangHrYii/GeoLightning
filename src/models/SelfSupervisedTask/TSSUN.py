import torch
import torch.nn as nn
import torch.nn.functional as F

class TSSUN(nn.Module):
    def __init__(self, num_classes=1000):
        super(TSSUN, self).__init__()
        self.num_classes = num_classes

    def forward(self, x):
        return x