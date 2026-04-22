# ShahedSpotter

Real-time UAV detection and tracking system built on YOLOv11n and ByteTrack. Designed to identify Shahed-class drones in live camera feeds or recorded video, with an audible alert on confirmed detection.

---

## Overview

The pipeline processes each frame through three stages:

```
capture → preprocess (WB + CLAHE) → YOLO detect + ByteTrack → HUD overlay + alert
```

**Detection classes**

| ID | Name | Description |
|----|------|-------------|
| 0 | `drone_other` | Non-Shahed UAVs |
| 1 | `not_drone` | Birds, planes, other false positives |
| 2 | `shahed` | Shahed-class target |

**Operating modes**

| Mode | Source | Use case |
|------|--------|----------|
| `live` | Camera device (USB / GStreamer / CSI) | Real-time monitoring |
| `forensic` | Video file (`file:<path>`) | Post-event analysis |

---

## Project Structure

```
ShahedSpotter/
├── config/
│   └── default.yaml            # Camera, detection, output settings
├── data/
│   └── dataset.yaml            # YOLO dataset split paths and class names
├── src/
│   ├── main.py                 # Pipeline orchestrator and entry point
│   ├── config.py               # Pydantic config models
│   ├── contracts.py            # Shared dataclasses (Frame, TrackedObject)
│   ├── capture/
│   │   └── video_capture.py    # Threaded frame producer (live + forensic)
│   ├── preprocessing/
│   │   └── image_enhance.py    # Gray-world WB + CLAHE enhancer
│   ├── tracking/
│   │   └── ultralytics_byte_tracker_adapter.py  # YOLO track() wrapper
│   └── output/
│       ├── hud_renderer.py     # OpenCV bounding-box overlay
│       └── alert_sound.py      # Audible siren on Shahed confirmation
├── training/
│   └── train_yolo.py           # Fine-tuning + evaluation script
└── requirements.txt
```

---

## Requirements

- Python 3.10+
- CUDA-capable GPU (recommended) or CPU

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Configuration

All settings are in `config/default.yaml`. The key sections:

```yaml
mode: "live"          # "live" | "forensic"

camera:
  source: 0           # device index for live, or "file:<path>" for forensic
  width: 640
  height: 480
  fps: 30
  intrinsics:         # camera calibration — required for azimuth/elevation output
    fx: 600.0
    fy: 600.0
    cx: 320.0
    cy: 240.0
  preprocessing:
    clahe_enabled: true
    clahe_clip_limit: 2.0
    clahe_tile_size: 8
    auto_white_balance: true

detection:
  model_path: "src/models/yolov11n_shahed_3class.pt"
  confidence_threshold: 0.30
  nms_iou_threshold: 0.45

output:
  overlay_enabled: true
```

---

## Running

**Live camera:**

```bash
python -m src.main --config config/default.yaml --preview
```

**Forensic (video file):**

Set `mode: "forensic"` and `source: "file:/path/to/video.mp4"` in `config/default.yaml`, then:

```bash
python -m src.main --config config/default.yaml
```

Press **Q** in the preview window or send `SIGINT` (`Ctrl+C`) to stop.

---

## Training

### 1. Prepare the dataset

Place your dataset under `data/` and fill in `data/dataset.yaml`:

```yaml
path: data/
train: images/train
val:   images/val
test:  images/test
nc: 3
names: ["drone_other", "not_drone", "shahed"]
```

### 2. Train

```bash
python training/train_yolo.py
```

Available options:

| Flag | Description |
|------|-------------|
| `--epochs N` | Number of training epochs (default: 100) |
| `--batch N` | Batch size (default: 8) |
| `--dry-run` | Validate config without training |
| `--eval-only WEIGHTS` | Skip training, run evaluation on given weights |
| `--export-only WEIGHTS` | Export weights to TensorRT or NCNN |
| `--format engine\|ncnn` | Export format (default: `engine`) |
| `--int8` | Apply INT8 quantisation during export |

### 3. Evaluate

Evaluation runs automatically after training. To run manually:

```bash
python training/train_yolo.py --eval-only src/models/yolov11n_shahed_3class.pt
```

Acceptance criteria checked:

| Metric | Threshold |
|--------|-----------|
| mAP@0.5 | ≥ 0.70 |
| shahed recall | ≥ 0.80 |
| not_drone precision | ≥ 0.80 |

Results are saved to `training/runs/eval/ac_report.json`.

### 4. Export for edge deployment

```bash
# TensorRT (Jetson)
python training/train_yolo.py --export-only src/models/yolov11n_shahed_3class.pt --format engine

# NCNN (Raspberry Pi)
python training/train_yolo.py --export-only src/models/yolov11n_shahed_3class.pt --format ncnn
```

---

## Preprocessing

`ImageEnhancer` (`src/preprocessing/image_enhance.py`) applies two deterministic steps to every frame **before** the model — both at training time and at inference time to ensure a consistent image distribution:

1. **Gray-world white balance** — scales each colour channel so its mean equals the global mean, neutralising colour casts from different lighting conditions.
2. **CLAHE** (Contrast Limited Adaptive Histogram Equalisation) — applied on the L channel of LAB colourspace to improve local contrast without affecting hue or saturation.

---

## Camera Support

`FrameProducer` tries two backends in order:

1. **GStreamer** — preferred on Jetson / Linux with hardware-accelerated capture.
2. **OpenCV VideoCapture** — fallback for Raspberry Pi, desktop, and CI environments.

On connection failure the producer waits 5 seconds and retries automatically.

---

## HUD

When `overlay_enabled: true`, each tracked object is drawn with:

- **Green** bounding box — `shahed` (threat)
- **Blue** bounding box — `drone_other` / `not_drone`
- Label: `ID:<id> <class> <confidence%>`
- FPS counter in the top-left corner
