import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.registry import register_fusion

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )
    def forward(self, x):
        return x * self.fc(self.pool(x))

@register_fusion("default_adapter")
class DetectionAwareFeatureAdapter(nn.Module):
    def __init__(self, sr_channels, det_channels, hidden_channels=None):
        super().__init__()
        hidden_channels = hidden_channels or det_channels
        self.proj = nn.Conv2d(sr_channels, hidden_channels, 1)
        self.depthwise = nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1, groups=hidden_channels, bias=False)
        self.pointwise = nn.Conv2d(hidden_channels, det_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(det_channels)
        self.act = nn.SiLU(inplace=True)
        self.ca = ChannelAttention(det_channels)
        self.residual = nn.Conv2d(sr_channels, det_channels, 1) if sr_channels != det_channels else None

    def forward(self, f_sr, out_size=None):
        if out_size is not None and f_sr.shape[-2:] != out_size:
            f_sr = F.interpolate(f_sr, size=out_size, mode="bilinear", align_corners=False)
        identity = self.residual(f_sr) if self.residual is not None else f_sr
        x = self.proj(f_sr)
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.ca(x)
        return x + identity
