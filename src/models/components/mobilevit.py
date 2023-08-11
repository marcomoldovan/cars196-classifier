import torch.nn as nn
import timm

class MobileViT(nn.Module):
    def __init__(self, num_classes, pretrained, dropout_value, freeze_layers):
        super().__init__()
        self.num_classes = num_classes
        self.mobilevit = timm.create_model(
            "mobilevit_s.cvnets_in1k", pretrained=pretrained, num_classes=0 # Set to 0 so we can add our own head later.
        )

        # Freezing layers if freeze_layers is True
        if freeze_layers:
            for param in self.mobilevit.parameters():
                param.requires_grad = False

        in_features = self.mobilevit.head.in_features
        self.head = nn.Sequential(
            nn.Linear(in_features, num_classes),
            nn.Dropout(dropout_value)
        )

    def forward(self, x):
        x = self.mobilevit(x)
        return self.head(x)
