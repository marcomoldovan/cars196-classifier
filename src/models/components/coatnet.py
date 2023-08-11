import torch.nn as nn
import timm

class CoaTNet(nn.Module):
    def __init__(self, num_classes, pretrained, freeze_layers, dropout_value):
        super().__init__()
        self.num_classes = num_classes
        self.coatnet = timm.create_model(
            "coat_lite_mini.in1k", pretrained=pretrained, num_classes=num_classes
        )

        if freeze_layers:
            for param in self.coatnet.parameters():
                param.requires_grad = False

        in_features = self.coatnet.head.in_features
        self.coatnet.head = nn.Sequential(
            nn.Linear(in_features, num_classes),
            nn.Dropout(dropout_value)
        )

    def forward(self, x):
        return self.coatnet(x)
