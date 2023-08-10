import torch.nn as nn
import timm


class EfficientNetv2(nn.Module):
    def __init__(self, num_classes, pretrained, freeze_layers):
        super().__init__()
        self.num_classes = num_classes
        self.efficientnet = timm.create_model(
            "tf_efficientnetv2_b2", pretrained=pretrained, num_classes=num_classes
        )

        # Freeze layers if specified
        if freeze_layers:
            for param in self.efficientnet.parameters():
                param.requires_grad = False

        in_features = self.efficientnet.classifier.in_features
        self.efficientnet.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.efficientnet(x)
