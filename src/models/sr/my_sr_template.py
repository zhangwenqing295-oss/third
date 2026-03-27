import torch.nn as nn
from src.models.registry import register_sr
from src.models.sr.base import BaseSRModel

@register_sr("my_sr")
class MySRTemplate(BaseSRModel):
    def __init__(self, **kwargs):
        super().__init__()
        self.stem = nn.Conv2d(3, 32, 3, 1, 1)
        self.body = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(32, 3, 3, 1, 1)

    def forward(self, x):
        f = self.body(self.stem(x))
        return self.head(f)

    def extract_features(self, x, stage="deep"):
        f0 = self.stem(x)
        f1 = self.body(f0)
        sr = self.head(f1)
        feat = f1 if stage == "deep" else f0
        return sr, feat
