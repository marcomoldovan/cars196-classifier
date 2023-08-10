import torch.nn as nn
import torchvision.models as models


class DenseNet161(nn.Module):
    def __init__(self, num_classes, pretrained, freeze_layers):
        super().__init__()
        self.num_classes = num_classes
        self.densenet = models.densenet161(pretrained=pretrained)

        # Freeze layers if specified
        if freeze_layers:
            for param in self.densenet.parameters():
                param.requires_grad = False

        in_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.densenet(x)
