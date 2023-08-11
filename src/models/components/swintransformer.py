import torch.nn as nn
import timm

class SwinTransformer(nn.Module):
    def __init__(self, num_classes, pretrained, dropout_value, freeze_layers):
        super().__init__()
        self.num_classes = num_classes
        self.swin = timm.create_model(
            "swin_s3_tiny_224.ms_in1k", pretrained=pretrained, num_classes=0 # Set to 0 so we can add our own head later.
        )

        # Freezing layers if freeze_layers is True
        if freeze_layers:
            for param in self.swin.parameters():
                param.requires_grad = False

        in_features = self.swin.head.in_features
        self.head = nn.Sequential(
            nn.Linear(in_features, num_classes),
            nn.Dropout(dropout_value)
        )

    def forward(self, x):
        x = self.swin(x)
        return self.head(x)
