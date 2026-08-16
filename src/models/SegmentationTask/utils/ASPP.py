import torch.nn as nn
import torch
from src.models.SegmentationTask.utils.bricks import BuildActivation, BuildNormalization

bn_mom = 0.0003

class ASPPModule(nn.ModuleList):
    """Atrous Spatial Pyramid Pooling (ASPP) Module.

    Args:
        dilations (tuple[int]): Dilation rate of each layer.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
    """

    def __init__(self, in_channels, out_channels, dilations = [1, 2, 4, 8], norm='layernorm'):
        super(ASPPModule, self).__init__()
        inter_channels = in_channels // 4
        self.aspp_blocks = nn.ModuleList([])
        for dilation in dilations:
            kernel_size = 1 if dilation == 1 else 3
            padding = 0 if dilation == 1 else dilation
            self.aspp_blocks.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        inter_channels,
                        kernel_size=kernel_size,
                        padding=padding,
                        stride=1,
                        dilation=dilation,
                    ),
                    BuildNormalization(norm, (inter_channels, {})),
                    nn.ReLU(inplace=True),
                )
            )
        self.conv_out = nn.Sequential(
                    torch.nn.Conv2d(inter_channels * len(dilations) + in_channels, out_channels, kernel_size=1, stride=1, padding=0),
                    BuildNormalization(norm, (out_channels, {})),
                    nn.ReLU(inplace=True),
                    )

    def forward(self, x):
        """Forward function."""
        aspp_outs = []
        for aspp_module in self.aspp_blocks:
            aspp_outs.append(aspp_module(x))
        aspp_outs.append(x)
        out = torch.cat(tuple(aspp_outs), 1)
        return self.conv_out(out)


__all__ = [
    "ASPPModule",
]