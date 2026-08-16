import torch.nn as nn
import torch
import torch.nn.functional as F
# from networks.utils.block import double_conv
bn_mom = 0.0003

class double_conv(torch.nn.Module):
    def __init__(
        self, in_chn, out_chn, stride=1, dilation=1
    ):  # params:in_chn(input channel of double conv),out_chn(output channel of double conv)
        super(double_conv, self).__init__()  ##parent's init func

        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, out_chn, kernel_size=3, stride=stride, dilation=dilation, padding=dilation),
            nn.BatchNorm2d(out_chn, momentum=bn_mom),
            torch.nn.ReLU(),
            torch.nn.Conv2d(out_chn, out_chn, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_chn, momentum=bn_mom),
            torch.nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class SEModule(nn.Module):   
    def __init__(self, channels, reduction_channels):
        super(SEModule, self).__init__()
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(
            channels, reduction_channels, kernel_size=1, padding=0, bias=True
        )
        self.ReLU = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(
            reduction_channels, channels, kernel_size=1, padding=0, bias=True
        )

    def forward(self, x):
        # x_se = self.avg_pool(x)
        x_se = x.view(x.size(0), x.size(1), -1).mean(-1).view(x.size(0), x.size(1), 1, 1)
        x_se = self.fc1(x_se)
        x_se = self.ReLU(x_se)
        x_se = self.fc2(x_se)
        return x * x_se.sigmoid()


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, downsample=None, use_se=False, stride=1, dilation=1):
        super(BasicBlock, self).__init__()

        first_planes = planes
        outplanes = planes * self.expansion

        self.conv1 = double_conv(inplanes, first_planes)
        self.conv2 = double_conv(
            first_planes, outplanes, stride=stride, dilation=dilation
        )
        self.se = SEModule(outplanes, planes // 4) if use_se else None
        self.downsample = (
            torch.nn.MaxPool2d(stride=2, kernel_size=2) if downsample else None
        )
        self.ReLU = nn.ReLU(inplace=True)

    def forward(self, x):

        out = self.conv1(x)
        residual = out
        out = self.conv2(out)

        if self.se is not None:
            out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out = out + residual
        out = self.ReLU(out)

        return out
