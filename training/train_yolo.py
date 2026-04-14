import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from loguru import logger
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = ROOT / "data" / "dataset.yaml"
MODELS_DIR = ROOT / "src" / "models"
RUNS_DIR = ROOT / "training"  / "runs"

# ── model ──────────────────────────────────────────────────────────────────────
MODEL_NAME = "yolo11n.pt"  # YOLOv11 nano — pretrained on COCO

# ── training hyperparameters ───────────────────────────────────────────────────
EPOCHS = 50
PATIENCE = 10 # early stopping
IMG_SIZE = 640
BATCH = 16              
WORKERS = 4 # number of CPU threads for data loading; set to 0 for single-threaded
DEVICE = "" # "" = auto-detect (cuda:0 / cpu)

# ── class info (matches data/dataset.yaml) ───────────────
CLASS_NAMES = ["drone_other", "not_drone", "shahed"]
NC = 3

# ── thresholds (used in post-eval report) ───────────────────────────────────
AC_MAP50_MIN = 0.80          
AC_SHAHED_RECALL_MIN = 0.85  
AC_NOT_DRONE_PREC_MIN = 0.80 

def build_augmentation_pipeline():
    """
    Applied in addition to Ultralytics built-in mosaic / mixup.
    Returns an A.Compose configured for bounding-box augmentation.
    """
    import albumentations as A

    return A.Compose(
        [
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5,
            ),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
            A.MotionBlur(blur_limit=(3, 7), p=0.3),
            A.GaussNoise(std_range=(5 / 255, 15 / 255), p=0.3),
            A.ImageCompression(quality_range=(50, 95), p=0.2),
            A.RandomFog(fog_coef_range=(0.1, 0.3), p=0.15),
            A.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.3,
                hue=0.05,
                p=0.2,
            ),
        ],
    )

def register_albumentations_callback(model, augment_pipeline) -> None:
    """
    Register a callback that applies the custom Albumentations pipeline
    on each training batch via Ultralytics' on_train_batch_start hook.
    """

    def on_train_batch_start(trainer):
        # Ultralytics stores the current batch in trainer.batch
        batch = getattr(trainer, "batch", None)
        if batch is None:
            return
        imgs = batch.get("img", None)
        if imgs is None:
            return

        import torch

        augmented = []
        for img_tensor in imgs:
            
            # Convert from tensors to numpy.
            # img_tensor: C×H×W float [0,1]
            img_np = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

            # TODO: Implement bboxes
            result = augment_pipeline(image=img_np)

            # Convert from numpy arrays to tensors.
            aug_np = result["image"].astype(np.float32) / 255.0
            augmented.append(torch.from_numpy(aug_np).permute(2, 0, 1))

        batch["img"] = torch.stack(augmented).to(imgs.device)

    model.add_callback("on_train_batch_start", on_train_batch_start)


def train(dry_run: bool = False) -> Path:
    if not DATA_YAML.exists():
        logger.error(f"Dataset YAML not found: {DATA_YAML}. Please check the path and try again.")
        sys.exit(1)

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading base model: {MODEL_NAME}")
    model = YOLO(MODEL_NAME)

    if dry_run:
        logger.info("Dry-run mode — skipping training. Config validated OK.")
        return Path("dry_run")

    augment = build_augmentation_pipeline()
    register_albumentations_callback(model, augment)

    logger.info(
        f"Starting training: {EPOCHS} epochs, patience={PATIENCE}, "
        f"batch={BATCH}, imgsz={IMG_SIZE}"
    )

    results = model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        patience=PATIENCE,
        imgsz=IMG_SIZE,
        batch=BATCH,
        workers=WORKERS,
        device=DEVICE,
        project=str(RUNS_DIR),
        name="train",
        exist_ok=True,
        # Ultralytics built-in augmentations (complement Albumentations)
        fliplr=0.5,
        degrees=5.0,
        translate=0.2,
        scale=0.3,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        mosaic=1.0,
        mixup=0.0,
        # Regularisation
        dropout=0.0,
        weight_decay=0.0005,
        warmup_epochs=3,
        # Logging
        plots=True,
        save=True,
        save_period=-1,  # only save best + last
        verbose=True,
    )

    best_weights = RUNS_DIR / "train" / "weights" / "best.pt"
    if best_weights.exists():
        dest = MODELS_DIR / "yolov11n_shahed_3class.pt"
        shutil.copy2(best_weights, dest)
        logger.info(f"Best weights copied to {dest}")
    else:
        logger.warning(f"Expected best.pt not found at {best_weights}")

    return best_weights

# ── evaluation ─────────────────────────────────────────────────────────────────

def evaluate(weights_path: Path) -> dict:
    """
    Runs validation on the test split, prints per-class AP + confusion matrix,
    and checks thresholds.
    Returns a results dict.
    """
    if not weights_path.exists():
        logger.error(f"Weights not found: {weights_path}")
        sys.exit(1)

    logger.info(f"Evaluating {weights_path} on test split ...")
    model = YOLO(str(weights_path))

    metrics = model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=IMG_SIZE,
        batch=BATCH,
        device=DEVICE,
        plots=True,
        verbose=True,
        project=str(RUNS_DIR),
        name="eval",
        exist_ok=True,
    )

    # Extract results
    map50 = float(metrics.box.map50)
    map50_95 = float(metrics.box.map)
    per_class_ap50 = metrics.box.ap50   # shape: (nc,)
    per_class_recall = metrics.box.r    # shape: (nc,)  (at conf threshold)
    per_class_precision = metrics.box.p # shape: (nc,)

    report = {
        "mAP@0.5": round(map50, 4),
        "mAP@0.5:0.95": round(map50_95, 4),
        "per_class": {},
    }

    logger.info("── Evaluation results ──────────────────────────────────")
    logger.info(f"  mAP@0.5       : {map50:.4f}  (AC-1.1 ≥ {AC_MAP50_MIN})")

    for i, name in enumerate(CLASS_NAMES):
        ap50 = float(per_class_ap50[i]) if i < len(per_class_ap50) else 0.0
        rec = float(per_class_recall[i]) if i < len(per_class_recall) else 0.0
        prec = float(per_class_precision[i]) if i < len(per_class_precision) else 0.0
        report["per_class"][name] = {
            "AP@0.5": round(ap50, 4),
            "recall": round(rec, 4),
            "precision": round(prec, 4),
        }
        logger.info(
            f"  [{i}] {name:<15} AP@0.5={ap50:.3f}  "
            f"recall={rec:.3f}  precision={prec:.3f}"
        )

    # AC checks
    logger.info("── AC checks ───────────────────────────────────────────")
    _ac(
        "AC-1.1", "mAP@0.5",
        map50, AC_MAP50_MIN,
    )
    shahed = report["per_class"]["shahed"]
    _ac(
        "AC-1.4", "shahed recall",
        shahed["recall"], AC_SHAHED_RECALL_MIN,
    )
    not_drone = report["per_class"]["not_drone"]
    _ac(
        "AC-1.5", "not_drone precision",
        not_drone["precision"], AC_NOT_DRONE_PREC_MIN,
    )

    # Save report
    report_path = RUNS_DIR / "eval" / "ac_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info(f"AC report saved: {report_path}")

    return report

def _ac(ac_id: str, metric_name: str, value: float, threshold: float) -> None:
    status = "PASS" if value >= threshold else "FAIL"
    logger.info(f"  {ac_id} [{status}] {metric_name} = {value:.4f} (threshold ≥ {threshold})")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ShahedSpotter YOLOv11n training")
    p.add_argument("--dry-run", action="store_true", help="Config check only")
    p.add_argument(
        "--eval-only",
        metavar="WEIGHTS",
        default=None,
        help="Skip training, evaluate given weights file",
    )
    p.add_argument(
        "--epochs", type=int, default=EPOCHS,
        help=f"Number of epochs (default {EPOCHS})",
    )
    p.add_argument(
        "--batch", type=int, default=BATCH,
        help=f"Batch size (default {BATCH})",
    )
    return p.parse_args()

def main() -> None:
    import torch
    print(torch.__version__)
    print(torch.version.cuda)
    print(torch.cuda.is_available())
    
    args = parse_args()

    global EPOCHS, BATCH
    EPOCHS = args.epochs
    BATCH = args.batch

    logger.info("=== ShahedSpotter — YOLOv11n training ===")
    logger.info(f"Dataset YAML : {DATA_YAML}")
    logger.info(f"Runs dir     : {RUNS_DIR}")
    logger.info(f"Models dir   : {MODELS_DIR}")

    if args.eval_only:
        weights = Path(args.eval_only)
        evaluate(weights)
        return

    best_weights = train(dry_run=args.dry_run)

    if args.dry_run:
        return

    if best_weights.exists():
        evaluate(best_weights)
    else:
        logger.warning("Best weights not found after training — skipping evaluation")

    logger.info("=== Training complete ===")

if __name__ == "__main__":
    main()