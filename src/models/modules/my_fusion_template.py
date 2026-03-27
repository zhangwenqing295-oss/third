import torch.nn as nn
import torch.nn.functional as F
from src.models.registry import register_fusion

@register_fusion("my_fusion")
class MyFusionTemplate(nn.Module):
    def __init__(self, sr_channels, det_channels, **kwargs):
        super().__init__()
        self.proj = nn.Conv2d(sr_channels, det_channels, 1)

    def forward(self, f_sr, out_size=None):
        if out_size is not None and f_sr.shape[-2:] != out_size:
            f_sr = F.interpolate(f_sr, size=out_size, mode="bilinear", align_corners=False)
        return self.proj(f_sr)
