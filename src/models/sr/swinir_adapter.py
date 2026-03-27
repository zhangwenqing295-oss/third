from pathlib import Path
import sys
import torch
from src.models.registry import register_sr
from src.models.sr.base import BaseSRModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
KAIR_ROOT = PROJECT_ROOT / "third_party" / "KAIR-master"
if str(KAIR_ROOT) not in sys.path:
    sys.path.insert(0, str(KAIR_ROOT))

from models.network_swinir import SwinIR  # noqa: E402

@register_sr("swinir")
class SwinIRAdapter(BaseSRModel):
    def __init__(self, weights="", **kwargs):
        super().__init__()
        self.net = SwinIR(**kwargs)
        if weights and Path(weights).exists():
            ckpt = torch.load(weights, map_location="cpu")
            state = ckpt.get("params_ema", ckpt.get("params", ckpt.get("state_dict", ckpt)))
            missing, unexpected = self.net.load_state_dict(state, strict=False)
            print(f"[SwinIR] load weights: missing={len(missing)}, unexpected={len(unexpected)}")

    def forward(self, x):
        return self.net(x)

    def extract_features(self, x, stage="deep"):
        # 基于 KAIR SwinIR 结构手动展开主路径
        x_first = self.net.conv_first(x)
        # 浅层/深层特征来自 deep feature extractor 之前和之后
        body_feat = self.net.forward_features(x_first)
        body_feat = self.net.conv_after_body(body_feat) + x_first

        # reconstruction path
        if self.net.upsampler == 'pixelshuffle':
            sr = self.net.conv_before_upsample(body_feat)
            sr = self.net.conv_last(self.net.upsample(sr))
        elif self.net.upsampler == 'pixelshuffledirect':
            sr = self.net.upsample(body_feat)
        elif self.net.upsampler == 'nearest+conv':
            sr = self.net.conv_before_upsample(body_feat)
            sr = self.net.lrelu(self.net.conv_up1(torch.nn.functional.interpolate(sr, scale_factor=2, mode='nearest')))
            if hasattr(self.net, 'conv_up2'):
                sr = self.net.lrelu(self.net.conv_up2(torch.nn.functional.interpolate(sr, scale_factor=2, mode='nearest')))
            sr = self.net.conv_last(self.net.lrelu(self.net.conv_hr(sr)))
        else:
            sr = x + self.net.conv_last(body_feat)

        feat = body_feat if stage == "deep" else x_first
        return sr, feat
