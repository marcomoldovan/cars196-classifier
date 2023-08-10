import torch.nn as nn
import torchvision.models as models


class ResNet50(nn.Module):
    def __init__(self, num_classes, pretrained, freeze_layers):
        super().__init__()
        self.num_classes = num_classes
        self.resnet = models.resnet50(pretrained=pretrained)

        # Freeze layers if specified
        if freeze_layers:
            for param in self.resnet.parameters():
                param.requires_grad = False

        in_features = self.resnet.fc.in_features

        self.resnet.fc = nn.Linear(in_features, num_classes)

        # self.resnet.fc = nn.Sequential(
        #     nn.Dropout(0.2),  # Apply dropout
        #     nn.Linear(in_features, num_classes)
        # )

        # self.resnet.fc = nn.Sequential(
        #     nn.Linear(self.resnet.fc.in_features, 512),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(512, num_classes),
        # )

    def forward(self, x):
        return self.resnet(x)
