import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import pytorch_lightning as pl


class ResNet34(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super(ResNet34, self).__init__()
        self.resnet34 = models.resnet34(pretrained)
        in_features = self.resnet34.fc.in_features
        self.resnet34.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.resnet34(x)
