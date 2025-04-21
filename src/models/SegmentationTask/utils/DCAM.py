import torch.nn as nn
import torch
import torch.nn.functional as F
from src.models.SegmentationTask.utils.block import *
bn_mom = 0.0003
"""
    DCAM: dual correlated attention module
    proposed in paper 'Object-level change detection with a dual correlation attention-guided detector'
"""

class CCAM(nn.Module):
    def __init__(self, channel):
        super(CCAM, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.conv_1x1 = nn.Conv2d(channel*2, channel, kernel_size=1, stride=1, padding=0)
        self.conv_3x3 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)

    def forward(self, f1, f2):
        m_batchsize, C, height, width = f1.size()
        f1_query = f1.view(m_batchsize, C, -1)                        #(B, W*H, C)
        f2_query = f2.view(m_batchsize, C, -1).permute(0, 2, 1)       #(B, C, W*H)
        s_coatt = self.softmax(torch.bmm(f1_query, f2_query))               #(B, C, C)

        fp1 = torch.bmm(s_coatt.permute(0, 2, 1), f1_query).view(m_batchsize, C, height, width)
        fp2 = torch.bmm(f2_query, s_coatt).view(m_batchsize, C, height, width)

        fp12 = torch.sigmoid(torch.cat((fp1, fp2), 1))
        wp = self.conv_1x1(fp12)
        wp = self.conv_3x3(wp)
        return torch.sigmoid(wp)


class PCAM(nn.Module):
    def __init__(self, channel):
        super(PCAM, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.conv_1x1 = nn.Conv2d(channel*2, channel, kernel_size=1, stride=1, padding=0)
        self.conv_3x3 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)

    def forward(self, f1, f2):
        m_batchsize, C, height, width = f1.size()
        f1_query = f1.view(m_batchsize, -1, width*height).permute(0, 2, 1)  #(B, W*H, C)
        f2_query = f2.view(m_batchsize, -1, width*height)                   #(B, C, W*H)
        s_coatt = self.softmax(torch.bmm(f1_query, f2_query))               #(B, W*H, W*H)
        fp1 = torch.bmm(s_coatt, f1_query).view(m_batchsize, C, height, width)
        fp2 = torch.bmm(f2_query, s_coatt.permute(0, 2, 1)).view(m_batchsize, C, height, width)

        fp12 = torch.sigmoid(torch.cat((fp1, fp2), 1))

        wp = self.conv_1x1(fp12)
        wp = self.conv_3x3(wp)
        return torch.sigmoid(wp)


class CDM(nn.Module):
    def __init__(self, channel):
        super(CDM, self).__init__()
        self.conv_1x1_1 = nn.Sequential(
            nn.Conv2d(channel*2, channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channel),
            nn.ReLU(),
        )
        self.conv_1x1_2 = nn.Sequential(
            nn.Conv2d(channel*2, channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channel),
            nn.ReLU(),
        )

    def forward(self, f1, f2):
        fd1 = f1 - f2
        f12 = torch.cat((f1, f2), 1)
        fd2 = self.conv_1x1_1(f12)
        fd = self.conv_1x1_2(torch.cat((fd1, fd2), 1))
        return fd

class DCAM(nn.Module):
    def __init__(self, channel):
        super(DCAM, self).__init__()
        self.CCAM = CCAM(channel)
        self.PCAM = PCAM(channel)
        self.CDM = CDM(channel)

        self.conv_out = nn.Sequential(
            nn.Conv2d(channel*4, channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channel),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(),
        )

    def forward(self, f1, f2):
        wp = self.PCAM(f1, f2)
        wc = self.CCAM(f1, f2)
        fd = self.CDM(f1, f2)
        f12 = torch.cat((f1, f2), 1)

        fp = torch.mul(fd, wp) + fd
        fc = torch.mul(fd, wc) + fd

        out = torch.cat((fp, fc, f12), 1)
        return self.conv_out(out)


__all__ = [
    "CCAM",
    "DCAM",
]