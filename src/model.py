import torch
from torch import nn
from torch.nn import functional as F


class vgg16(nn.Module):
    def __init__(self):
        super.__init__(vgg16, self)
        self.conv_1 = nn.Conv2d(3, 64, 3)
        self.conv_2 = nn.Conv2d(64, 64, 3)
        self.maxpool_1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=2)
        self.conv_3 = nn.Conv2d(64, 128, 3)
        self.conv_4 = nn.Conv2d(128, 128, 3)
        self.maxpool_2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=2)
        self.conv_5 = nn.Conv2d(128, 256, 3)
        self.conv_6 = nn.Conv2d(256, 256, 3)
        self.conv_7 = nn.Conv2d(256, 256, 3)
        self.maxpool_3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=2)
        self.conv_8 = nn.Conv2d(256, 512, 3)
        self.conv_9 = nn.Conv2d(512, 512, 3)
        self.maxpool_4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=2)
        self.conv_9 = nn.Conv2d(512, 512, 3)
        self.conv_10 = nn.Conv2d(512, 512, 3)
        self.conv_11 = nn.Conv2d(512, 512, 3)
        self.maxpool_5 = nn.MaxPool2d(kernel_size=2, stride=2, padding=2)




