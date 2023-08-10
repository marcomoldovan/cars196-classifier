import torch.nn as nn
import timm


class SwinTransformer(nn.Module):
    def __init__(self, num_classes, pretrained):
        super().__init__()
        self.num_classes = num_classes
        self.swin = timm.create_model(
            "swin_s3_tiny_224.ms_in1k",
            pretrained=pretrained,
            num_classes=num_classes,
        )
        # in_features = self.swin.head.in_features
        # self.swin.head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.swin(x)
