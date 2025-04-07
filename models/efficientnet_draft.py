# TODO: Replace this with actual EfficientNet or ConvNeXt implementation.
import torch.nn as nn


class EfficientNetDraft(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),
        )

    def forward(self, x):
        return self.model(x)
