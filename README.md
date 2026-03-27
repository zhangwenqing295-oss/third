# Integrated KAIR + YOLOv8 Workspace

这个工作目录已经把你上传的两份源码整合进来了：

- `src/third_party/KAIR-master/`
- `src/third_party/yolov8_repo/`

默认训练时优先使用本地源码，而不是 site-packages 里的版本。

## 快速开始

```bash
pip install -r requirements.txt
python tools/init_experiment.py --name exp01 --config configs/default.yaml
python train.py --config configs/default.yaml --print-yolo-layers
python train.py --config configs/default.yaml
```

## 关键位置

- 本地 YOLOv8 源码：`src/third_party/yolov8_repo/ultralytics/`
- 本地 KAIR 源码：`src/third_party/KAIR-master/models/network_swinir.py`
- SwinIR 适配器：`src/models/sr/swinir_adapter.py`
- YOLOv8 适配器：`src/models/det/yolov8_adapter.py`
- 两阶段模型：`src/models/two_stage_model.py`

## 后续替换自己的方法

你只需要替换下面三个模块即可：
- `src/models/sr/my_sr_template.py`
- `src/models/det/my_detector_template.py`
- `src/models/modules/my_fusion_template.py`
