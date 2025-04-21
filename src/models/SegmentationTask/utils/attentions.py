import torch.nn as nn
import torch
import torch.nn.functional as F

bn_mom = 0.0003

class Spatial_SelfAttention(nn.Module):
    def __init__(self, in_channels, key_channels, out_channels, scale=1, bn_type=None):
        super(Spatial_SelfAttention, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = key_channels
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_key = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.key_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(self.key_channels),
        )
        self.f_query = self.f_key
        self.f_value = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.value_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.W = nn.Conv2d(
            in_channels=self.value_channels,
            out_channels=self.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)
        value = self.f_value(x).view(batch_size, self.value_channels, -1)
        value = value.permute(0, 2, 1).contiguous()  # B * (H*W) * value_channels
        query = self.f_query(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1).contiguous()  # B * (H*W) * key_channels
        key = self.f_key(x).view(
            batch_size, self.key_channels, -1
        )  # B * key_channels * (H*W)

        sim_map = torch.matmul(query, key)  # B * (H*W) * (H*W)
        sim_map = (self.key_channels ** -0.5) * sim_map  # B * (H*W) * (H*W)
        sim_map = torch.softmax(sim_map, dim=-1)  # B * (H*W) * (H*W)

        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.value_channels, *x.size()[2:])
        context = self.W(context)
        if self.scale > 1:
            if torch.__version__ == "0.4":
                context = F.upsample(
                    input=context, size=(h, w), mode="bilinear", align_corners=True
                )
        return context


class SAM(nn.Module):
    def __init__(self, channel):
        super(SAM, self).__init__()
        self.squeeze = nn.Conv2d(channel, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        z = self.squeeze(x)
        z = self.sigmoid(z)
        return z


class CAM(nn.Module):
    def __init__(self, input_dim, reduction=4):
        super(CAM, self).__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.global_maxpool = nn.AdaptiveMaxPool2d(1)
        self.conv1 = nn.Conv2d(input_dim, input_dim // reduction, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(input_dim // reduction, input_dim, kernel_size=1, stride=1)
        self.conv3 = nn.Conv2d(input_dim, input_dim // reduction, kernel_size=1, stride=1)
        self.conv4 = nn.Conv2d(input_dim // reduction, input_dim, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        z1 = self.global_avgpool(x)
        z2 = self.global_maxpool(x)
        z1 = self.relu(self.conv1(z1))
        z1 = self.conv2(z1)
        z2 = self.relu(self.conv3(z2))
        z2 = self.conv4(z2)
        return self.sigmoid(z1 + z2)


class PAM_Module(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out


class CAM_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim


        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out


class Change_CAM(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Change_CAM, self).__init__()
        self.out_channel = out_channel
        self.proj_qkv = nn.Conv2d(in_channel, in_channel*3, kernel_size=1, stride=1, padding=0)
        
        self.conv_merge = nn.Sequential(
            nn.Conv2d(in_channel*2, out_channel, kernel_size=1, stride=1, dilation=1, padding=0),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x1, x2):
        bs, C, height, width = x1.size()

        qkv1 = self.proj_qkv(x1).view(bs, -1, 3, height * width).permute(2, 0, 1, 3)
        q1, k1, v1 = qkv1[0], qkv1[1], qkv1[2]  #B, C, H*W
        qkv2 = self.proj_qkv(x2).view(bs, -1, 3, height * width).permute(2, 0, 1, 3)
        q2, k2, v2 = qkv2[0], qkv2[1], qkv2[2]  #B, C, H*W

        k = torch.cat((k1, k2), 1)
        v = torch.cat((v1, v2), 1)

        energy_C1 = torch.bmm(q1, k.transpose(1, 2))  # B * C * 2C
        energy_C1 = energy_C1 * (self.out_channel) ** -0.5
        attention_C1 = torch.softmax(energy_C1, dim=-1)
        energy_C2 = torch.bmm(q2, k.transpose(1, 2))  # B * C * 2C
        energy_C2 = energy_C2 * (self.out_channel) ** -0.5
        attention_C2 = torch.softmax(energy_C2, dim=-1)
        x1 = x1 + self.gamma * torch.bmm(attention_C1, v).view(bs, C, height, width)  # B * C * H * W
        x2 = x2 + self.gamma * torch.bmm(attention_C2, v).view(bs, C, height, width)  # B * C * H * W
        y = self.conv_merge(torch.cat((x1, x2), 1))  # B * C * H * W
        return x1, x2, y