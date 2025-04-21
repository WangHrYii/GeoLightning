#
import torch.nn as nn
import torch
import torch.nn.functional as F
bn_mom = 0.0003

class MFIM(nn.Module):
    """
    Implemention of Multi-scale feature Interaction Module
    """
    def __init__(self, channels):
        super(MFIM, self).__init__()
        """
        Args:
            channels: A list of integer corresponding to the channel numbers of input features.
                    The list is in the ascending order.
        """
        self.channels_blocks = channels
        self.num_base_layers = len(self.channels_blocks)
        self.downsample_blocks = nn.ModuleList([])
        self.upsample_blocks = nn.ModuleList([])
        self.fuse_blocks = nn.ModuleList([])
        self.relu = nn.ReLU()
        for i in range(self.num_base_layers):
            self.fuse_blocks.append(
                nn.Sequential(
                    torch.nn.Conv2d(self.channels_blocks[i], self.channels_blocks[i], kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(self.channels_blocks[i], momentum=bn_mom),
                    nn.ReLU(inplace=True),
                    )
            )
            if i > 0:
                self.downsample_blocks.append(
                     nn.Sequential(
                        nn.MaxPool2d(kernel_size=2, stride=2),
                        torch.nn.Conv2d(self.channels_blocks[i-1], self.channels_blocks[i], kernel_size=1, stride=1, padding=0),
                        nn.BatchNorm2d(self.channels_blocks[i], momentum=bn_mom),
                        )
                )
            if i < self.num_base_layers - 1:
                self.upsample_blocks.append(
                     nn.Sequential(
                        nn.ConvTranspose2d(self.channels_blocks[i+1], self.channels_blocks[i], kernel_size=4, stride=2, padding=1),
                        nn.BatchNorm2d(self.channels_blocks[i], momentum=bn_mom),
                        )
                )

    def forward(self, features):
        """
        Args:
            features: A list of features with different stage from the encoder. 
                    Asuming the size of i-th feature as (h, w), the size of (i+1)-th feature is (h/2, w/2).
        Output:
            out_features: A list of features with the same element shape as the input.
        
        """
        out_features = []
        for i in range(self.num_base_layers):
            if i == 0:
                out_features.append(self.fuse_blocks[i](self.relu(features[i] + self.upsample_blocks[i](features[i+1]))))
            elif i == self.num_base_layers - 1:
                out_features.append(self.fuse_blocks[i](self.relu(features[i] + self.downsample_blocks[i-1](features[i-1]))))
            else:
                out_features.append(self.fuse_blocks[i](self.relu(features[i] + self.upsample_blocks[i](features[i+1])) + \
                        self.downsample_blocks[i-1](features[i-1])))
        return out_features


class MFIM2(nn.Module):
    """
    Implemention of Multi-scale feature Interaction Module, use concat and residul to fuse features
    """
    def __init__(self, channels):
        super(MFIM2, self).__init__()
        """
        Args:
            channels: A list of integer corresponding to the channel numbers of input features.
                    The list is in the ascending order.
        """
        self.channels_blocks = channels
        self.num_base_layers = len(self.channels_blocks)
        self.downsample_blocks = nn.ModuleList([])
        self.upsample_blocks = nn.ModuleList([])
        self.fuse_blocks = nn.ModuleList([])
        for i in range(self.num_base_layers):
            if i > 0 and i < self.num_base_layers - 1:
                self.fuse_blocks.append(
                    nn.Sequential(
                        torch.nn.Conv2d(self.channels_blocks[i]*3, self.channels_blocks[i], kernel_size=1, stride=1, padding=0),
                        nn.BatchNorm2d(self.channels_blocks[i], momentum=bn_mom),
                        nn.ReLU()
                        )
                )
            else:
                self.fuse_blocks.append(
                    nn.Sequential(
                        torch.nn.Conv2d(self.channels_blocks[i]*2, self.channels_blocks[i], kernel_size=1, stride=1, padding=0),
                        nn.BatchNorm2d(self.channels_blocks[i], momentum=bn_mom),
                        nn.ReLU()
                        )
                )
            if i > 0:
                self.downsample_blocks.append(
                     nn.Sequential(
                        torch.nn.Conv2d(self.channels_blocks[i-1], self.channels_blocks[i], kernel_size=3, stride=2, padding=1),
                        nn.BatchNorm2d(self.channels_blocks[i], momentum=bn_mom),
                        nn.ReLU()
                        )
                )
            if i < self.num_base_layers - 1:
                self.upsample_blocks.append(
                     nn.Sequential(
                        nn.Upsample(scale_factor=2, mode="nearest"),
                        torch.nn.Conv2d(self.channels_blocks[i+1], self.channels_blocks[i], kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(self.channels_blocks[i], momentum=bn_mom),
                        nn.ReLU()
                        )
                )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, features):
        """
        Args:
            features: A list of features with different stage from the encoder. 
                    Asuming the size of i-th feature as (h, w), the size of (i+1)-th feature is (h/2, w/2).
        Output:
            out_features: A list of features with the same element shape as the input.
        
        """
        out_features = []
        for i in range(self.num_base_layers):
            if i == 0:
                out_features.append(self.fuse_blocks[i](torch.cat((features[i], self.upsample_blocks[i](features[i+1])), 1)))
            elif i == self.num_base_layers - 1:
                out_features.append(self.fuse_blocks[i](torch.cat((features[i], self.downsample_blocks[i-1](features[i-1])), 1)))
            else:
                out_features.append(self.fuse_blocks[i](torch.cat((features[i], self.upsample_blocks[i](features[i+1]), \
                        self.downsample_blocks[i-1](features[i-1])), 1)))
        return out_features


class MFIM3(nn.Module):
    """
    Implemention of Multi-scale feature Interaction Module.
    Use conv3x3 with stride=2 to downsample features.
    Use upsample bilinear and conv1x1 to upsample features.
    """
    def __init__(self, channels):
        super(MFIM3, self).__init__()
        """
        Args:
            channels: A list of integer corresponding to the channel numbers of input features.
                    The list is in the ascending order.
        """
        self.channels_blocks = channels
        self.num_base_layers = len(self.channels_blocks)
        self.fuse_layers = nn.ModuleList([])
        self.fuse_blocks = nn.ModuleList([])
        self.relu = nn.ReLU()
        for i in range(self.num_base_layers):
            self.fuse_blocks.append(
                nn.Sequential(
                    torch.nn.Conv2d(self.channels_blocks[i], self.channels_blocks[i], kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(self.channels_blocks[i], momentum=bn_mom),
                    )
            )
            fuse_layer = nn.ModuleList([])
            for j in range(self.num_base_layers):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(
                                self.channels_blocks[j],
                                self.channels_blocks[i],
                                3,
                                1,
                                1,
                                bias=False,
                            ),
                            nn.BatchNorm2d(self.channels_blocks[i]),
                            nn.Upsample(scale_factor=2 ** (j - i), mode="nearest"),
                        )
                    )
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = nn.ModuleList([])
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = self.channels_blocks[i]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        self.channels_blocks[j],
                                        num_outchannels_conv3x3,
                                        3,
                                        2,
                                        1,
                                        bias=False,
                                    ),
                                    nn.BatchNorm2d(
                                        num_outchannels_conv3x3
                                    ),
                                )
                            )
                        else:
                            num_outchannels_conv3x3 = self.channels_blocks[j]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        self.channels_blocks[j],
                                        num_outchannels_conv3x3,
                                        3,
                                        2,
                                        1,
                                        bias=False,
                                    ),
                                    nn.BatchNorm2d(
                                        num_outchannels_conv3x3
                                    ),
                                    nn.ReLU(False),# downsample gradually, do relu in the middle level
                                )
                            )
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            self.fuse_layers.append(nn.ModuleList(fuse_layer))

    def forward(self, features):
        """
        Args:
            features: A list of features with different stage from the encoder. 
                    Asuming the size of i-th feature as (h, w), the size of (i+1)-th feature is (h/2, w/2).
        Output:
            out_features: A list of features with the same element shape as the input.
        
        """
        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = features[0] if i == 0 else self.fuse_layers[i][0](features[0])
            for j in range(1, self.num_base_layers):
                if i == j:
                    y = y + features[j]
                else:
                    y = y + self.fuse_layers[i][j](features[j])
            x_fuse.append(self.relu(self.fuse_blocks[i](self.relu(y))))

        return x_fuse