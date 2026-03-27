from pathlib import Path
import sys

# 使用本地 YOLOv8 源码里的 trainer
PROJECT_ROOT = Path(__file__).resolve().parents[1]
YOLO_REPO = PROJECT_ROOT / "third_party" / "yolov8_repo"
if str(YOLO_REPO) not in sys.path:
    sys.path.insert(0, str(YOLO_REPO))

from ultralytics.models.yolo.detect import DetectionTrainer  # noqa: E402
from src.models.two_stage_model import TwoStageYOLOv8

class SRDATrainer(DetectionTrainer):
    def __init__(self, *args, cfg_dict=None, **kwargs):
        self.cfg_dict = cfg_dict or {}
        super().__init__(*args, **kwargs)

    def get_model(self, cfg=None, weights=None, verbose=True):
        model_cfg = self.cfg_dict["model"]
        loss_cfg = self.cfg_dict["loss"]
        data_yaml = self.cfg_dict["data"]["yaml"]

        wrapper = TwoStageYOLOv8(
            det_weights=model_cfg["det"]["weights"],
            data_yaml=data_yaml,
            sr_cfg=model_cfg["sr"],
            fusion_cfg=model_cfg["fusion"],
            fuse_layer=model_cfg.get("fuse_layer", 15),
            sr_feature_stage=model_cfg.get("sr_feature_stage", "deep"),
            lambda_sr=loss_cfg.get("lambda_sr", 0.0),
            lambda_align=loss_cfg.get("lambda_align", 0.1),
            freeze_sr=model_cfg.get("freeze_sr", True),
            verbose=verbose,
        )

        model = wrapper.model
        def predict_with_sr(x, profile=False, visualize=False, augment=False, embed=None):
            return wrapper.predict(x, profile=profile, visualize=visualize, augment=augment, embed=embed)
        model.predict = predict_with_sr

        def loss_with_sr(batch, preds=None):
            return wrapper.loss(batch, preds=preds)
        model.loss = loss_with_sr
        return model
