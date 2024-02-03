from unicodedata import numeric
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter
from torchvision import transforms
from math import log
from torch.nn.utils import weight_norm
from timm.models.layers import DropPath
import math
import os
import numpy as np


class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, gamma=2, b=1):
        super(eca_layer, self).__init__()
        t = int(abs((log(channel, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=int(k/2), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, l]
        b, c, l = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.transpose(-1, -2)).transpose(-1, -2)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class Block(nn.Module):
    def __init__(self, dim, scale = 4):
        super().__init__()

        self.scales = scale
        self.group = dim // scale
        self.conv2 = nn.ModuleList([nn.Conv1d(self.group, self.group, kernel_size=3, padding=1,
                                                groups=self.group) for _ in range(scale - 1)])
        self.bn2 = nn.ModuleList([nn.BatchNorm1d(self.group) for _ in range(scale - 1)])

        # self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv group=dim
        # self.norm = nn.BatchNorm1d(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.SELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        # self.se = SELayer(channel=dim)
        self.eca = eca_layer(channel=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x

        xs = torch.chunk(x, self.scales, 1)  # 将x分割成scales块
        ys = []
        for s in range(self.scales):
            if s == 0:
                ys.append(xs[s])
            elif s == 1:
                ys.append(self.act(self.bn2[s - 1](self.conv2[s - 1](xs[s]))))
            else:
                ys.append(self.act(self.bn2[s - 1](self.conv2[s - 1](xs[s] + ys[-1]))))
        x = torch.cat(ys, 1)

        # x = self.dwconv(x)
        # x = self.norm(x)
        x = x.permute(0, 2, 1)  # [N, C, M] -> [N, M, C]
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 2, 1)  # [N, M, C] -> [N, C, M]
        
        # x = shortcut + x
        x = shortcut + self.eca(x)
        return x


class ConvNeXt(nn.Module):
    def __init__(self, in_chans: int = 1, num_classes: int = 2, drop_path_rate: float = 0.1, depths: list = None, mkernel_size: int = 1,
                 dims: list = None):
        super().__init__()
        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(nn.Conv1d(in_chans, dims[0], kernel_size=7, padding=3),
                             nn.BatchNorm1d(dims[0]))
        self.downsample_layers.append(stem)

        # 对应stage2-stage4前的3个downsample
        for i in range(3):
            downsample_layer = nn.Sequential(nn.BatchNorm1d(dims[i]),
                                             nn.Conv1d(dims[i], dims[i+1], kernel_size=7, padding=3),
                                             nn.MaxPool1d(kernel_size= 9))
            self.downsample_layers.append(downsample_layer)


        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple blocks
        # 构建每个stage中堆叠的block
        for i in range(4):
            stage = nn.Sequential(
                    *[Block(dim=dims[i])
                        for j in range(depths[i])]
                )
            self.stages.append(stage)

        self.norm = nn.BatchNorm1d(dims[-1])
        self.head = nn.Linear(32, num_classes)
        self.ln1 = nn.Linear(128,64)
        self.ln2 = nn.Linear(64,32)
        self.ac = nn.SELU()

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        x = self.ac(self.norm(x))
        # print(x.shape)
        x = F.max_pool1d(x, x.shape[-1])
        x = torch.flatten(x, start_dim=1)
        # print(x.shape)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.ac(self.ln1(x))
        x = self.ac(self.ln2(x))
        x = self.head(x)
        return x

def convnext_1d_tiny(num_classes: int):
    #
    model = ConvNeXt(depths=[1, 2, 3, 1],
                     dims=[16, 32, 64, 128],
                     num_classes=num_classes)
    return model


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = convnext_1d_tiny(num_classes=2)
    x = torch.randn(2, 1, 96000)
    output = model(x)
    print(output.shape)
    print(output)


