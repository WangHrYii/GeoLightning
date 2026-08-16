import torch.nn as nn
import torch
from src.models.SegmentationTask.utils.attentions import Spatial_SelfAttention

bn_mom = 0.0003
class FPN(nn.Module):
    """ feature parymid network"""

    def __init__(self, in_dim):
        super(FPN, self).__init__()

        self.conv_e1 = nn.Sequential(
            nn.Conv2d(in_dim, in_dim * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_dim * 2, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.conv_e2 = nn.Sequential(
            nn.Conv2d(in_dim * 2, in_dim * 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_dim * 4, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.conv_e3 = nn.Sequential(
            nn.Conv2d(in_dim * 4, in_dim * 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_dim * 8, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.conv_d1 = nn.Sequential(
            nn.Conv2d(in_dim * 2, in_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_dim, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.conv_d2 = nn.Sequential(
            nn.Conv2d(in_dim * 4, in_dim * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_dim * 2, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.conv_d3 = nn.Sequential(
            nn.Conv2d(in_dim * 8, in_dim * 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_dim * 4, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.downsample_x2 = nn.MaxPool2d(stride=2, kernel_size=2)
        self.upsample_x2 = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x):
        e1 = self.downsample_x2(self.conv_e1(x))
        e2 = self.downsample_x2(self.conv_e2(e1))
        e3 = self.downsample_x2(self.conv_e3(e2))

        d3 = self.upsample_x2(self.conv_d3(e3))
        d2 = self.upsample_x2(self.conv_d2(e2 + d3))
        d1 = self.upsample_x2(self.conv_d1(e1 + d2))
        return d1


class FPN_SSA(nn.Module):
    """ feature parymid network"""

    def __init__(self, in_dim):
        super(FPN_SSA, self).__init__()

        self.conv_e1 = nn.Sequential(
            nn.Conv2d(in_dim, in_dim * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_dim * 2, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.conv_e2 = nn.Sequential(
            nn.Conv2d(in_dim * 2, in_dim * 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_dim * 4, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.conv_e3 = nn.Sequential(
            nn.Conv2d(in_dim * 4, in_dim * 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_dim * 8, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.conv_d1 = nn.Sequential(
            nn.Conv2d(in_dim * 2, in_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_dim, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.conv_d2 = nn.Sequential(
            nn.Conv2d(in_dim * 4, in_dim * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_dim * 2, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.conv_d3 = nn.Sequential(
            nn.Conv2d(in_dim * 8, in_dim * 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_dim * 4, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.ssa3 = Spatial_SelfAttention(in_dim * 4, in_dim, in_dim * 4)
        self.ssa2 = Spatial_SelfAttention(in_dim * 2, in_dim, in_dim * 2)
        self.ssa1 = Spatial_SelfAttention(in_dim, in_dim, in_dim)

        self.downsample_x2 = nn.MaxPool2d(stride=2, kernel_size=2)
        self.upsample_x2 = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x):
        e1 = self.downsample_x2(self.conv_e1(x))
        e2 = self.downsample_x2(self.conv_e2(e1))
        e3 = self.downsample_x2(self.conv_e3(e2))

        d3 = self.conv_d3(e3)
        ssa = self.ssa3(d3)
        d3 = self.upsample_x2(d3 * ssa)
        d2 = self.conv_d2(e2 + d3)
        ssa = self.ssa2(d2)
        d2 = self.upsample_x2(d2 * ssa)
        d1 = self.conv_d1(e1 + d2)
        ssa = self.ssa1(d1)
        d1 = self.upsample_x2(d1 * ssa)

        return d1


class NL_Block(nn.Module):
    def __init__(self, in_channels):
        super(NL_Block, self).__init__()
        self.conv_v = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
        )
        self.W = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        batch_size, c, h, w = x.size(0),x.size(1), x.size(2), x.size(3)
        value = self.conv_v(x).view(batch_size, c, -1)
        value = value.permute(0, 2, 1)                                             #B * (H*W) * value_channels
        key = x.view(batch_size, c, -1)                #B * key_channels * (H*W)
        query = x.view(batch_size, c, -1)
        query = query.permute(0, 2, 1)  
        sim_map = torch.matmul(query, key)                                         #B * (H*W) * (H*W)
        sim_map = (c**-.5) * sim_map                               #B * (H*W) * (H*W)
        sim_map = torch.softmax(sim_map, dim=-1)                                       #B * (H*W) * (H*W)
        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, c, *x.size()[2:])
        context = self.W(context)
        
        return context


class NL_FPN(nn.Module):
    """ non-local feature parymid network"""
    def __init__(self, in_dim, reduction=True):
        super(NL_FPN, self).__init__()
        if reduction:
            self.reduction = nn.Sequential(
                        nn.Conv2d(in_dim, in_dim//4, kernel_size=1, stride=1, padding=0),
                        nn.BatchNorm2d(in_dim//4, momentum=bn_mom),
                        nn.ReLU(inplace=True),
                        )
            self.re_reduction = nn.Sequential(
                        nn.Conv2d(in_dim//4, in_dim, kernel_size=1, stride=1, padding=0),
                        nn.BatchNorm2d(in_dim, momentum=bn_mom),
                        nn.ReLU(inplace=True),
                        )
            in_dim = in_dim//4
        else:
            self.reduction = None
            self.re_reduction = None
        self.conv_e1 = nn.Sequential(
                        nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(in_dim, momentum=bn_mom),
                        nn.ReLU(inplace=True),
                        )
        self.conv_e2 = nn.Sequential(
                        nn.Conv2d(in_dim, in_dim*2,kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(in_dim*2, momentum=bn_mom),
                        nn.ReLU(inplace=True),
                        )
        self.conv_e3 = nn.Sequential(
                        nn.Conv2d(in_dim*2,in_dim*4,kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(in_dim*4, momentum=bn_mom),
                        nn.ReLU(inplace=True),
                        )
        self.conv_d1 = nn.Sequential(
                        nn.Conv2d(in_dim,in_dim,kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(in_dim, momentum=bn_mom),
                        nn.ReLU(inplace=True),
                        )
        self.conv_d2 = nn.Sequential(
                        nn.Conv2d(in_dim*2,in_dim,kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(in_dim, momentum=bn_mom),
                        nn.ReLU(inplace=True),
                        )
        self.conv_d3 = nn.Sequential(
                        nn.Conv2d(in_dim*4,in_dim*2,kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(in_dim*2, momentum=bn_mom),
                        nn.ReLU(inplace=True),
                        )
        self.nl3 = NL_Block(in_dim*2)
        self.nl2 = NL_Block(in_dim)
        self.nl1 = NL_Block(in_dim)

        self.downsample_x2 = nn.MaxPool2d(stride=2, kernel_size=2)
        self.upsample_x2 = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x):
        if self.reduction is not None:
            x = self.reduction(x)
        e1 = self.conv_e1(x)                            #C,H,W
        e2 = self.conv_e2(self.downsample_x2(e1))       #2C,H/2,W/2
        e3 = self.conv_e3(self.downsample_x2(e2))       #4C,H/4,W/4
        
        d3 = self.conv_d3(e3)                           #2C,H/4,W/4
        nl = self.nl3(d3)
        d3 = self.upsample_x2(torch.mul(d3, nl))                  ##2C,H/2,W/2
        d2 = self.conv_d2(e2+d3)                        #C,H/2,W/2
        nl = self.nl2(d2)
        d2 = self.upsample_x2(torch.mul(d2, nl))                 #C,H,W
        d1 = self.conv_d1(e1+d2)
        nl = self.nl1(d1)
        d1 = torch.mul(d1, nl)          #C,H,W
        if self.re_reduction is not None:
            d1 = self.re_reduction(d1)

        return d1
