import torch.nn as nn
import torch
import torch.nn.functional as F
from src.models.SegmentationTask.utils.se_basicblock import SEModule
from src.models.SegmentationTask.utils.bricks import BuildNormalization

bn_mom = 0.0003

class base_conv(torch.nn.Module):
    def __init__(
        self, in_chn, out_chn, kernel_size=3, stride=1, dilation=1, padding=1
    ):  # params:in_chn(input channel of double conv),out_chn(output channel of double conv)
        super(base_conv, self).__init__()  ##parent's init func

        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_chn,
                out_chn,
                kernel_size=3,
                stride=1,
                dilation=dilation,
                padding=dilation,
            ),
            nn.BatchNorm2d(out_chn, momentum=bn_mom),
            torch.nn.ReLU()
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class cat(torch.nn.Module):
    def __init__(self, in_chn_high, in_chn_low, out_chn, upsample=False):
        super(cat, self).__init__()  ##parent's init func
        self.do_upsample = upsample
        self.upsample = torch.nn.Upsample(
            scale_factor=2, mode="nearest"
        )
        self.conv2d = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_chn_high + in_chn_low, out_chn, kernel_size=1, stride=1, padding=0
            ),
            nn.BatchNorm2d(out_chn, momentum=bn_mom),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x, y):
        if self.do_upsample:
            x = self.upsample(x)
        x = torch.cat(
            (x, y), 1
        )  # x,y shape(batch_sizxe,channel,w,h), concat at the dim of channel
        return self.conv2d(x)


class cat_conv(cat):
    def __init__(self, in_chn_high, in_chn_low, out_chn, upsample=False):
        super(cat_conv, self).__init__(in_chn_high, in_chn_low, out_chn, upsample)  ##parent's init func
        self.conv2d_2 = torch.nn.Sequential(
            torch.nn.Conv2d(out_chn, out_chn, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_chn, momentum=bn_mom),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x, y):
        if self.do_upsample:
            x = self.upsample(x)
        out = torch.cat(
            (x, y), 1
        )  # x,y shape(batch_sizxe,channel,w,h), concat at the dim of channel
        out = self.conv2d(out)
        return self.conv2d_2(out) + out

class bridge(torch.nn.Module):
    def __init__(self, in_chn):
        super(bridge, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_chn, in_chn * 2, kernel_size=3, stride=2, padding=1
            ),
            nn.BatchNorm2d(in_chn * 2),
            torch.nn.ReLU(),
        )
        self.conv2 = torch.nn.Sequential(
            nn.ConvTranspose2d(in_chn * 2, in_chn, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(in_chn),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        diffY = x.size()[2] - y.size()[2]
        diffX = x.size()[3] - y.size()[3]

        y = F.pad(y, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        return self.relu(y + x)


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch,residual=True):
        super(DoubleConv, self).__init__()
        self.residual = residual
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu2 = nn.ReLU(inplace=True)


    def forward(self, input):
        x0 = self.conv1(input)
        x0 = self.bn1(x0)
        x0=self.relu1(x0)
        x = self.conv2(x0)
        x = self.bn2(x)
        x = self.relu2(x)
        if self.residual:
            x=x+x0
        return x


class SeDoubleConv(DoubleConv):
    def __init__(self, in_ch, out_ch, residual=True, use_se=True):
        super(SeDoubleConv, self).__init__(in_ch, out_ch,residual)
        self.se = SEModule(out_ch, out_ch//4) if use_se else None

    def forward(self, input):
        x0 = self.conv1(input)
        x0 = self.bn1(x0)
        x0=self.relu1(x0)
        x = self.conv2(x0)
        x = self.bn2(x)
        x = self.relu2(x)
        if self.se is not None:
            x = self.se(x)
        if self.residual:
            x = x + x0
        return x

class AlgebraFuse(nn.Module):
    """
        use algebra (add and diff) to fuse features
    """
    def __init__(self, in_channel, norm='layernorm'):
        super(AlgebraFuse, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel*2, in_channel, kernel_size=1, stride=1, dilation=1, padding=0),
            BuildNormalization('layernorm', (in_channel, {})),
            nn.ReLU(),
        )
    
    def forward(self, x1, x2):
        add = (x1 + x2)/2
        diff = x1 - x2
        y = torch.cat((add, diff), 1)
        y = self.conv(y)
        return y