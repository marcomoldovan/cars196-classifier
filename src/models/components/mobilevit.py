import torch.nn as nn
import timm


class MobileViT(nn.Module):
    def __init__(self, num_classes, pretrained):
        super().__init__()
        self.num_classes = num_classes
        self.mobilevit = timm.create_model(
            "mobilevit_s.cvnets_in1k", pretrained=pretrained, num_classes=num_classes
        )

    def forward(self, x):
        return self.mobilevit(x)
