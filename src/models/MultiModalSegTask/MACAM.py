import torch
import torch.nn as nn
import torch.nn.functional as F

class MCAM(nn.Module):
    def __init__(self, in_channels):
        super(MCAM, self).__init__()

        self.in_channels = in_channels
        # self.inter_channels = inter_channels if inter_channels else in_channels // 2

        # 1x1卷积层，用于从输入特征图中提取VOPT、QOPT和KOPT
        self.VOPT_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.QOPT_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.KOPT_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # 1x1卷积层，用于从输入特征图中提取VSAR、QSAR和KSAR
        self.VSAR_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.QSAR_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.KSAR_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, sar_features, optical_features):
        batch_size = optical_features.size(0)

        # 提取光学图像特征的VOPT、QOPT和KOPT
        VOPT = self.VOPT_conv(optical_features)
        QOPT = self.QOPT_conv(optical_features)
        KOPT = self.KOPT_conv(optical_features)

        # 提取SAR图像特征的VSAR、QSAR和KSAR
        VSAR = self.VSAR_conv(sar_features)
        QSAR = self.QSAR_conv(sar_features)
        KSAR = self.KSAR_conv(sar_features)

        # 计算光学图像特征的自注意力图Shigh_OPT
        S_OPT = torch.softmax(torch.matmul(QOPT.permute(0, 1, 3, 2), KOPT), dim=-1)

        # 计算SAR图像特征的自注意力图Shigh_SAR
        S_SAR = torch.softmax(torch.matmul(QSAR.permute(0, 1, 3, 2), KSAR), dim=-1)

        # 计算光学和SAR图像特征的交叉融合注意力图Shigh_cro
        S_cro = S_OPT * S_SAR

        # 使用光学特征图VOPT加权得到注意力加权后的光学特征图Atthigh_OPT
        Att_OPT = S_cro * VOPT

        # 使用SAR特征图VSAR加权得到注意力加权后的SAR特征图Atthigh_SAR
        Att_SAR = S_cro * VSAR

        # 将注意力加权后的光学和SAR特征图进行叠加得到最终的联合注意力图Atthigh_OPT_SAR
        Att_OPT_SAR = Att_OPT * Att_SAR

        return Att_OPT_SAR


if __name__ == '__main__':
    model = MCAM(in_channels=256)
    model.train()
    sar = torch.randn(2, 256, 64, 64)
    opt = torch.randn(2, 256, 64, 64)
    print(model)
    print("input:", sar.shape, opt.shape)
    print("output:", model(sar, opt).shape)