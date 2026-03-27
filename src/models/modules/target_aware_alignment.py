import torch
import torch.nn as nn
import torch.nn.functional as F

class TargetAwareAlignment(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, f_adapted, f_encoded):
        if f_adapted.shape[-2:] != f_encoded.shape[-2:]:
            f_adapted = F.interpolate(f_adapted, size=f_encoded.shape[-2:], mode="bilinear", align_corners=False)
        g = self.gate(torch.cat([f_adapted, f_encoded], dim=1))
        return g * f_adapted + (1.0 - g) * f_encoded
