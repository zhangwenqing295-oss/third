import argparse
import json
import shutil
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent / "src" / "third_party" / "yolov8_repo"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO  # noqa: E402
from src.utils.config import load_yaml, ensure_dir
from src.trainers.srda_trainer import SRDATrainer

def print_yolo_layers(weights):
    model = YOLO(weights).model
    print("\n[YOLOv8 layer index helper]")
    for i, m in enumerate(model.model):
        print(f"{i:>3d}: {m.__class__.__name__}")
    print()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--print-yolo-layers", action="store_true")
    return parser.parse_args()

def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    if args.print_yolo_layers:
        print_yolo_layers(cfg["model"]["det"]["weights"])
        return

    exp_dir = Path(cfg["project"]) / cfg["name"]
    ensure_dir(exp_dir)
    shutil.copy2(args.config, exp_dir / "config_used.yaml")
    with open(exp_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    train_cfg = cfg["train"]
    overrides = {
        "model": cfg["model"]["det"]["weights"],
        "data": cfg["data"]["yaml"],
        "epochs": train_cfg["epochs"],
        "imgsz": train_cfg["imgsz"],
        "batch": train_cfg["batch"],
        "workers": train_cfg["workers"],
        "device": train_cfg["device"],
        "project": cfg["project"],
        "name": cfg["name"],
        "lr0": train_cfg["lr0"],
        "lrf": train_cfg["lrf"],
        "optimizer": train_cfg["optimizer"],
        "weight_decay": train_cfg["weight_decay"],
        "close_mosaic": train_cfg["close_mosaic"],
        "amp": train_cfg["amp"],
        "pretrained": train_cfg["pretrained"],
        "task": "detect",
        "save": True,
        "exist_ok": True,
    }
    trainer = SRDATrainer(overrides=overrides, cfg_dict=cfg)
    trainer.train()

if __name__ == "__main__":
    main()
