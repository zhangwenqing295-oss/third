import torch
import torch.nn.functional as F
import yaml

from src.models.registry import build_sr, build_fusion
from src.models.modules.feature_encoder import DetectionFeatureEncoder
from src.models.modules.target_aware_alignment import TargetAwareAlignment
from src.utils.masks import boxes_to_mask
from src.models.det.yolov8_adapter import YOLOv8Builder

import src.models.sr.swinir_adapter
import src.models.sr.my_sr_template
import src.models.modules.detection_aware_adapter
import src.models.modules.my_fusion_template

def load_dataset_nc(data_yaml: str) -> int:
    with open(data_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if isinstance(data.get("names"), list):
        return len(data["names"])
    return int(data["nc"])

class TwoStageYOLOv8:
    def __init__(self, det_weights, data_yaml, sr_cfg, fusion_cfg, fuse_layer=15,
                 sr_feature_stage="deep", lambda_sr=0.0, lambda_align=0.1,
                 freeze_sr=True, verbose=True):
        nc = load_dataset_nc(data_yaml)
        self.builder = YOLOv8Builder(det_weights, nc=nc, verbose=verbose)
        self.model = self.builder.model

        sr_name = sr_cfg.get("name", "swinir")
        sr_kwargs = dict(sr_cfg.get("kwargs", {}))
        if "weights" in sr_cfg:
            sr_kwargs["weights"] = sr_cfg["weights"]
        self.sr_branch = build_sr(sr_name, **sr_kwargs)

        self.fusion_cfg = fusion_cfg
        self.fuse_layer = int(fuse_layer)
        self.sr_feature_stage = sr_feature_stage
        self.lambda_sr = float(lambda_sr)
        self.lambda_align = float(lambda_align)
        self.freeze_sr = freeze_sr

        if freeze_sr:
            for p in self.sr_branch.parameters():
                p.requires_grad = False
            self.sr_branch.eval()

        self.adapter = None
        self.encoder = None
        self.aligner = None
        self._modules_initialized = False

        self._cached_sr_img = None
        self._cached_sr_feat = None
        self._cached_encoded_feat = None
        self._cached_aligned_feat = None

        self._hook_handle = None
        self._register_hook()
        self.model._two_stage_wrapper = self

    def _register_hook(self):
        target_module = self.model.model[self.fuse_layer]
        def _hook(module, inputs, output):
            if self._cached_sr_feat is None or not isinstance(output, torch.Tensor):
                return output
            if not self._modules_initialized:
                det_c = output.shape[1]
                sr_c = self._cached_sr_feat.shape[1]
                fusion_name = self.fusion_cfg.get("name", "default_adapter")
                fusion_kwargs = dict(self.fusion_cfg.get("kwargs", {}))
                self.adapter = build_fusion(fusion_name, sr_channels=sr_c, det_channels=det_c, **fusion_kwargs).to(output.device)
                self.encoder = DetectionFeatureEncoder(in_channels=det_c, embed_channels=det_c).to(output.device)
                self.aligner = TargetAwareAlignment(channels=det_c).to(output.device)
                self._modules_initialized = True
            adapted = self.adapter(self._cached_sr_feat, out_size=output.shape[-2:])
            encoded = self.encoder(output)
            aligned = self.aligner(adapted, encoded)
            self._cached_encoded_feat = encoded
            self._cached_aligned_feat = aligned
            return output + aligned
        self._hook_handle = target_module.register_forward_hook(_hook)

    def predict(self, x, profile=False, visualize=False, augment=False, embed=None):
        if self.freeze_sr:
            with torch.no_grad():
                sr_img, sr_feat = self.sr_branch.extract_features(x, stage=self.sr_feature_stage)
        else:
            sr_img, sr_feat = self.sr_branch.extract_features(x, stage=self.sr_feature_stage)
        self._cached_sr_img = sr_img
        self._cached_sr_feat = sr_feat
        return self.model.predict(x, profile=profile, visualize=visualize, augment=augment, embed=embed)

    def loss(self, batch, preds=None):
        if preds is None:
            preds = self.predict(batch["img"])
        det_loss, det_items = self.model.criterion(preds, batch)

        sr_loss = torch.zeros((), device=batch["img"].device)
        if self.lambda_sr > 0 and "hr_img" in batch and self._cached_sr_img is not None:
            hr = batch["hr_img"]
            sr = self._cached_sr_img
            if sr.shape[-2:] != hr.shape[-2:]:
                sr = F.interpolate(sr, size=hr.shape[-2:], mode="bilinear", align_corners=False)
            sr_loss = F.l1_loss(sr, hr)

        align_loss = torch.zeros((), device=batch["img"].device)
        if self._cached_aligned_feat is not None and self._cached_encoded_feat is not None:
            f_a = self._cached_aligned_feat
            f_h = self._cached_encoded_feat.detach()
            mask = boxes_to_mask(batch["batch_idx"], batch["bboxes"], (f_a.shape[-2], f_a.shape[-1]), f_a.shape[0], f_a.device)
            diff = (f_a - f_h) ** 2
            align_loss = diff.mean() if mask.sum() < 1 else (diff * mask).sum() / (mask.sum() * f_a.shape[1] + 1e-6)

        total_loss = det_loss + self.lambda_sr * sr_loss + self.lambda_align * align_loss
        if isinstance(det_items, torch.Tensor):
            extra = torch.tensor([sr_loss.detach().item(), align_loss.detach().item()], device=det_items.device, dtype=det_items.dtype)
            det_items = torch.cat([det_items, extra], dim=0)
        return total_loss, det_items
