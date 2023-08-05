import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class FireBlock(nn.Module):
    def __init__(
        self, in_channels, squeeze_channels, expand1x1_channels, expand3x3_channels
    ):
        super(FireBlock, self).__init__()
        self.squeeze = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
        self.expand1x1 = nn.Conv2d(squeeze_channels, expand1x1_channels, kernel_size=1)
        self.expand3x3 = nn.Conv2d(
            squeeze_channels, expand3x3_channels, kernel_size=3, padding=1
        )

    def forward(self, x):
        x = F.relu(self.squeeze(x))
        out1x1 = F.relu(self.expand1x1(x))
        out3x3 = F.relu(self.expand3x3(x))
        return torch.cat([out1x1, out3x3], 1)


class SqueezeNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=7, stride=2)
        self.fire2 = FireBlock(96, 16, 64, 64)
        self.fire3 = FireBlock(128, 16, 64, 64)
        self.fire4 = FireBlock(128, 32, 128, 128)
        self.fire5 = FireBlock(256, 32, 128, 128)
        self.fire6 = FireBlock(256, 48, 192, 192)
        self.fire7 = FireBlock(384, 48, 192, 192)
        self.fire8 = FireBlock(384, 64, 256, 256)
        self.fire9 = FireBlock(512, 64, 256, 256)
        self.dropout = nn.Dropout(p=0.5)
        self.conv10 = nn.Conv2d(512, num_classes, kernel_size=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=3, stride=2, ceil_mode=True)

        x = self.fire2(x)
        x = self.fire3(x)
        x = self.fire4(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, ceil_mode=True)

        x = self.fire5(x)
        x = self.fire6(x)
        x = self.fire7(x)
        x = self.fire8(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, ceil_mode=True)

        x = self.fire9(x)
        x = self.dropout(x)
        x = self.conv10(x)
        x = F.avg_pool2d(x, kernel_size=13)
        x = torch.flatten(x, 1)
        return x
