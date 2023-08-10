import torch.nn as nn
import torchvision.models as models


class DenseNet161(nn.Module):
    def __init__(self, num_classes, pretrained, freeze_layers, dropout_value):
        super().__init__()
        self.num_classes = num_classes
        self.densenet = models.densenet161(pretrained=pretrained)

        # Freeze layers if specified
        if freeze_layers:
            for param in self.densenet.parameters():
                param.requires_grad = False

        # Set requires_grad to True for the parameters of the last few layers
        for (
            param
        ) in (
            self.densenet.features.denseblock4.denselayer24.parameters()
        ):  # Modify based on your desired layers
            param.requires_grad = True

        in_features = self.densenet.classifier.in_features
        # self.densenet.classifier = nn.Linear(in_features, num_classes)

        self.densenet.classifier = nn.Sequential(
            nn.Dropout(dropout_value), nn.Linear(in_features, num_classes)  # Apply dropout
        )


    def forward(self, x):
        return self.densenet(x)
