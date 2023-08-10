import torch.nn as nn
import timm


class EfficientNetv2(nn.Module):
    def __init__(self, num_classes, pretrained, freeze_layers, num_unfrozen_layers, dropout_value):
        super().__init__()
        self.num_classes = num_classes
        self.efficientnet = timm.create_model(
            "tf_efficientnetv2_b2", pretrained=pretrained, num_classes=num_classes
        )

        # Freeze layers if specified
        if freeze_layers:
            for param in self.efficientnet.parameters():
                param.requires_grad = False

        # Set requires_grad to True for the parameters of the last few layers
        unfrozen_layers = self.efficientnet.blocks[
            -num_unfrozen_layers:
        ]  # Get the last few blocks
        for block in unfrozen_layers:
            for param in block.parameters():
                param.requires_grad = True

        in_features = self.efficientnet.classifier.in_features
        # self.efficientnet.classifier = nn.Linear(in_features, num_classes)

        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(dropout_value), nn.Linear(in_features, num_classes)  # Apply dropout
        )

    def forward(self, x):
        return self.efficientnet(x)
