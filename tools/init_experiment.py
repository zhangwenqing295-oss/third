from pathlib import Path
import argparse
import shutil

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--root", type=str, default="experiments")
    args = parser.parse_args()

    root = Path(args.root) / args.name
    (root / "configs").mkdir(parents=True, exist_ok=True)
    (root / "weights").mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    (root / "outputs").mkdir(parents=True, exist_ok=True)
    cfg_dst = root / "configs" / Path(args.config).name
    shutil.copy2(args.config, cfg_dst)
    (root / "README.txt").write_text(
        "Experiment folder created.\n"
        "- configs/: copied config\n"
        "- weights/: pretrained checkpoints\n"
        "- logs/: training logs\n"
        "- outputs/: predictions and visualizations\n",
        encoding="utf-8",
    )
    print(f"initialized experiment at: {root}")
    print(f"copied config to: {cfg_dst}")

if __name__ == "__main__":
    main()
