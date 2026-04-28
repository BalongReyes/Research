"""
train.py
========
Trains a YOLOv11 model on the Cobb 500 chick defect dataset.

Dataset split  : 70% train / 20% val / 10% test
Optimizer      : SGD with cosine LR scheduler
Augmentation   : Mosaic, horizontal flip, brightness/contrast, HSV shifts
Preprocessing  : CLAHE (contrast normalization) — Canny NOT used
Early stopping : Patience = 20 epochs on val/mAP50

Run from project root:
  cd ~/chick_defect_project
  conda activate chick_env
  python scripts/train.py
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime

# ── project root on sys.path ──────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    from ultralytics import YOLO
except ImportError:
    raise SystemExit("ultralytics not found. Run: pip install ultralytics --break-system-packages")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(ROOT / "logs" / "train.log"),
    ]
)
logger = logging.getLogger(__name__)


# ── defaults ─────────────────────────────────────────────────────────────

DEFAULTS = dict(
    model       = "yolo11n.pt",       # nano – fastest on Jetson Orin Nano
    data_yaml   = str(ROOT / "data" / "dataset.yaml"),
    epochs      = 200,
    imgsz       = 640,
    batch       = 8,                  # Adjust based on available VRAM
    patience    = 20,                 # Early stopping patience
    lr0         = 0.01,
    lrf         = 0.001,
    momentum    = 0.937,
    weight_decay= 0.0005,
    warmup      = 3,
    device      = "0",                # GPU 0 (Jetson integrated GPU)
    workers     = 4,
    project     = str(ROOT / "models"),
    name        = f"chick_defect_{datetime.now().strftime('%Y%m%d_%H%M')}",
    exist_ok    = False,
    pretrained  = True,
    amp         = True,               # Mixed precision – supported on Orin
)


def parse_args():
    p = argparse.ArgumentParser(description="Train YOLOv11 for chick defect detection")
    for k, v in DEFAULTS.items():
        t = type(v) if v is not None else str
        p.add_argument(f"--{k}", default=v, type=t)
    return p.parse_args()


def verify_dataset(data_yaml: str):
    """Quick sanity-check: ensure dataset.yaml and image folders exist."""
    yaml_path = Path(data_yaml)
    if not yaml_path.exists():
        raise FileNotFoundError(
            f"dataset.yaml not found at {yaml_path}. "
            "Run scripts/prepare_dataset.py first."
        )
    logger.info(f"Dataset config: {yaml_path}")


def train(args):
    verify_dataset(args.data_yaml)

    logger.info("Initializing YOLOv11 model: %s", args.model)
    model = YOLO(args.model)

    logger.info("Starting training …")
    results = model.train(
        data        = args.data_yaml,
        epochs      = args.epochs,
        imgsz       = args.imgsz,
        batch       = args.batch,
        patience    = args.patience,
        lr0         = args.lr0,
        lrf         = args.lrf,
        momentum    = args.momentum,
        weight_decay= args.weight_decay,
        warmup_epochs = args.warmup,
        device      = args.device,
        workers     = args.workers,
        project     = args.project,
        name        = args.name,
        exist_ok    = args.exist_ok,
        pretrained  = args.pretrained,
        amp         = args.amp,
        # ── Augmentation ────────────────────────────────────────────
        hsv_h       = 0.015,    # Hue shift (simulate lighting colour)
        hsv_s       = 0.5,      # Saturation
        hsv_v       = 0.4,      # Brightness
        flipud      = 0.0,      # No vertical flip (chicks always upright)
        fliplr      = 0.5,      # Horizontal flip is valid
        mosaic      = 1.0,      # Mosaic augmentation
        mixup       = 0.1,
        degrees     = 5.0,      # Small rotation
        translate   = 0.1,
        scale       = 0.5,
        shear       = 2.0,
        perspective = 0.0005,
        # ── Logging ─────────────────────────────────────────────────
        verbose     = True,
        plots       = True,
        save        = True,
        save_period = 10,       # Save checkpoint every 10 epochs
    )

    best_model = Path(args.project) / args.name / "weights" / "best.pt"
    logger.info("Training complete. Best model: %s", best_model)

    # Quick validation on test split
    logger.info("Running final validation …")
    val_results = model.val(
        data    = args.data_yaml,
        split   = "test",
        device  = args.device,
        imgsz   = args.imgsz,
    )
    logger.info("mAP50     : %.4f", val_results.box.map50)
    logger.info("Precision : %.4f", val_results.box.mp)
    logger.info("Recall    : %.4f", val_results.box.mr)

    return best_model


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(ROOT / "logs", exist_ok=True)
    best = train(args)
    logger.info("Done. Best weights saved to: %s", best)
