from pathlib import Path
import sys
import copy

def import_local_ultralytics():
    project_root = Path(__file__).resolve().parents[2]
    repo_root = project_root / "third_party" / "yolov8_repo"
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel
    return YOLO, DetectionModel

class YOLOv8Builder:
    def __init__(self, weights, nc, verbose=True, **kwargs):
        YOLO, DetectionModel = import_local_ultralytics()
        self.YOLO = YOLO
        self.DetectionModel = DetectionModel
        self.base_yolo = YOLO(weights)
        base_model = self.base_yolo.model
        base_cfg = copy.deepcopy(base_model.yaml)
        self.model = DetectionModel(cfg=base_cfg, ch=3, nc=nc, verbose=verbose)
        self.model.load_state_dict(base_model.state_dict(), strict=False)
