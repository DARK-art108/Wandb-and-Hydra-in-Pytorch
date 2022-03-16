import torch
from torch import nn
from torch.nn import functional as F

class NetModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv_2 = nn.Conv2d(32, 64, 3)
        self.maxpool_1 = nn.MaxPool2d(2,2)
        self.conv_3 = nn.Conv2d(64, 128, 3)
        self.bn = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(21632, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv_1(x))
        x = F.relu(self.conv_2(x))
        x = self.maxpool_1(x)
        x = F.relu(self.conv_3(x))
        x = self.bn(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Below Network Require some large computation means you should have enough GPU memory with CUDA enabled to train from scratch else you can prefer transfer learning modules and timm.
# class VGG16_NET(nn.Module):
#     def __init__(self):
#         super(VGG16_NET, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)

#         self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
#         self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)

#         self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
#         self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
#         self.conv7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

#         self.conv8 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
#         self.conv9 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
#         self.conv10 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)

#         self.conv11 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
#         self.conv12 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
#         self.conv13 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)

#         self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

#         self.fc14 = nn.Linear(512, 4096)
#         self.fc15 = nn.Linear(4096, 4096)
#         self.fc16 = nn.Linear(4096, 10)

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = self.maxpool(x)
#         x = F.relu(self.conv3(x))
#         x = F.relu(self.conv4(x))
#         x = self.maxpool(x)
#         x = F.relu(self.conv5(x))
#         x = F.relu(self.conv6(x))
#         x = F.relu(self.conv7(x))
#         x = self.maxpool(x)
#         x = F.relu(self.conv8(x))
#         x = F.relu(self.conv9(x))
#         x = F.relu(self.conv10(x))
#         x = self.maxpool(x)
#         x = F.relu(self.conv11(x))
#         x = F.relu(self.conv12(x))
#         x = F.relu(self.conv13(x))
#         x = self.maxpool(x)
#         #print(x.shape)
#         x = torch.flatten(x, 1)
#         x = F.relu(self.fc14(x))
#         x = F.dropout(x, 0.5)
#         x = F.relu(self.fc15(x))
#         x = F.dropout(x, 0.5)
#         x = self.fc16(x)
#         return x


