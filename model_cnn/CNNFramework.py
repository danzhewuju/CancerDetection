#!/usr/bin/python
'''
CNN的整体网络框架结构
'''
import torch.nn as nn
from .fig import *


class ConvNet(nn.Module):
    def __init__(self, num_classes=num_classes):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)

        )
        self.fc1 = nn.Linear(6 * 6 * 32, 32)
        self.fc2 = nn.Linear(32, 8)
        self.fc3 = nn.Linear(8, num_classes)  # 取决于最后的个数种类

    def forward(self, x):
        # x = x.view(x.size(0), -1)
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.reshape(out.size(0), -1)  # 这里面的-1代表的是自适应的意思。
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out


# vgg16, 可以看到, 带有参数的刚好为16个
net_arch16 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M5', "FC1", "FC2",
              "FC"]

# vgg19, 基本和 vgg16 相同, 只不过在后3个卷积段中, 每个都多了一个卷积层
net_arch19 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M5',
              "FC1", "FC2", "FC"]  # VGG19的网络参数暂时有问题


class VGGNet(nn.Module):
    def __init__(self, num_classes):
        # net_arch 即为上面定义的列表: net_arch16 或 net_arch19
        super(VGGNet, self).__init__()
        self.num_classes = num_classes
        layers = []
        in_channels = 3  # 初始化通道数
        for arch in net_arch19:
            if arch == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            elif arch == 'M5':
                layers.append(nn.MaxPool2d(kernel_size=3, stride=1, padding=1))
            elif arch == "FC1":
                layers.append(nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=6, dilation=6))
                layers.append(nn.ReLU(inplace=True))
            elif arch == "FC2":
                layers.append(nn.Conv2d(1024, 1024, kernel_size=1))
                layers.append(nn.ReLU(inplace=True))
            elif arch == "FC":
                # layers.append(nn.Conv2d(1024, 1024, kernel_size=1))
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                layers.append(nn.Linear(3 * 3 * 1024, self.num_classes))
            else:
                layers.append(nn.Conv2d(in_channels=in_channels, out_channels=arch, kernel_size=3, padding=1))
                layers.append(nn.ReLU(inplace=True))
                in_channels = arch
                self.vgg = nn.ModuleList(layers)

    def forward(self, input_data):
        x = input_data
        for layer in self.vgg:
            x = layer(x)
        x = x.reshape(x.size(0), -1)
        out = x
        return out
