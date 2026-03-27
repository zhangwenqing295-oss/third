#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
import shutil
import sys

def main():
    if len(sys.argv) != 3 or sys.argv[1] != "--dst":
        print("usage: python generate_integrated_workspace_from_uploaded_sources.py --dst ./your_workspace")
        raise SystemExit(1)

    src = Path(__file__).resolve().parent / "project"
    dst = Path(sys.argv[2]) / "project"
    if dst.exists():
        raise FileExistsError(f"{dst} already exists")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)
    print(f"workspace generated at: {dst}")
    print("next:")
    print(f"  cd {dst}")
    print("  pip install -r requirements.txt")
    print("  python tools/init_experiment.py --name exp01 --config configs/default.yaml")
    print("  python train.py --config configs/default.yaml --print-yolo-layers")
    print("  python train.py --config configs/default.yaml")

if __name__ == "__main__":
    main()
