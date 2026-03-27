import torch.nn as nn

class DetectionFeatureEncoder(nn.Module):
    def __init__(self, in_channels, embed_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, embed_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.SiLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)
