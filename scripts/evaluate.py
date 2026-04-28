"""
evaluate.py
===========
Evaluates the trained YOLOv11 model on the test split and generates:
  - Confusion matrix (per Table 2 in the research paper)
  - Accuracy, Precision, Recall, F1-Score (per-class and overall)
  - mAP@50 and mAP@50-95
  - CSV report saved to logs/eval_report_<timestamp>.csv

Usage:
  conda activate chick_env
  python scripts/evaluate.py --model models/<run>/weights/best.pt

Optional:
  --data    path to dataset.yaml   (default: data/dataset.yaml)
  --imgsz   inference image size   (default: 640)
  --conf    confidence threshold   (default: 0.40)
  --iou     NMS IoU threshold      (default: 0.45)
  --device  cuda or cpu            (default: cuda)
"""

import sys
import argparse
import logging
import csv
from pathlib import Path
from datetime import datetime

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    from ultralytics import YOLO
except ImportError:
    raise SystemExit("ultralytics not found. Run: pip install ultralytics --break-system-packages")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CLASS_NAMES = [
    "Normal",
    "Eye Abnormality",
    "Deformed Beak",
    "Knock Knees",
    "Split Legs",
    "Bend Knees",
    "Crooked Toes",
]


# ── confusion matrix helpers ─────────────────────────────────────────────

def compute_binary_cm(tp: int, fp: int, fn: int, tn: int) -> dict:
    """Return accuracy, precision, recall, F1 from binary CM values."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    accuracy  = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0
    return dict(accuracy=accuracy, precision=precision, recall=recall, f1=f1,
                tp=tp, fp=fp, fn=fn, tn=tn)


def plot_confusion_matrix(matrix: np.ndarray, class_names: list, save_path: Path):
    """Save a heatmap of the normalised confusion matrix."""
    if not HAS_PLOT:
        logger.warning("matplotlib/seaborn not installed – skipping confusion matrix plot.")
        return

    norm = matrix.astype(float)
    row_sums = norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    norm = norm / row_sums

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax,
        linewidths=0.5, linecolor="#cccccc",
    )
    ax.set_xlabel("Predicted Class", fontsize=12)
    ax.set_ylabel("Actual Class",    fontsize=12)
    ax.set_title("Normalised Confusion Matrix – Cobb 500 Chick Defect Detection", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Confusion matrix saved → {save_path}")


def plot_per_class_bar(metrics_per_class: list, save_path: Path):
    """Bar chart of Precision / Recall / F1 per class."""
    if not HAS_PLOT:
        return
    names = [m["class"] for m in metrics_per_class]
    prec  = [m["precision"] for m in metrics_per_class]
    rec   = [m["recall"]    for m in metrics_per_class]
    f1    = [m["f1"]        for m in metrics_per_class]

    x = np.arange(len(names))
    w = 0.25
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - w, prec, w, label="Precision", color="#2980B9")
    ax.bar(x,     rec,  w, label="Recall",    color="#27AE60")
    ax.bar(x + w, f1,   w, label="F1-Score",  color="#E74C3C")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Per-Class Detection Metrics")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Per-class bar chart saved → {save_path}")


# ── evaluation logic ─────────────────────────────────────────────────────

def evaluate(model_path: str, data_yaml: str, imgsz: int,
             conf: float, iou: float, device: str):

    logs_dir = ROOT / "logs"
    logs_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger.info(f"Loading model: {model_path}")
    model = YOLO(model_path)

    # ── Run validation on test split ──────────────────────────────────────
    logger.info("Running inference on test split …")
    results = model.val(
        data    = data_yaml,
        split   = "test",
        imgsz   = imgsz,
        conf    = conf,
        iou     = iou,
        device  = device,
        plots   = True,
        save_json = False,
        verbose = True,
    )

    # ── Extract per-class metrics from ultralytics results ────────────────
    nc = len(CLASS_NAMES)
    metrics_per_class = []

    # results.box exposes per-class arrays when classes is not None
    ap50_per_class  = results.box.ap50   if hasattr(results.box, "ap50")  else [None]*nc
    prec_per_class  = results.box.p      if hasattr(results.box, "p")     else [None]*nc
    rec_per_class   = results.box.r      if hasattr(results.box, "r")     else [None]*nc

    for i, name in enumerate(CLASS_NAMES):
        p  = float(prec_per_class[i]) if i < len(prec_per_class) else 0.0
        r  = float(rec_per_class[i])  if i < len(rec_per_class)  else 0.0
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        ap = float(ap50_per_class[i]) if ap50_per_class[i] is not None else 0.0
        metrics_per_class.append(dict(class_name=name, class_id=i,
                                      precision=p, recall=r, f1=f1, ap50=ap))

    # ── Binary confusion matrix (Normal vs Defective) ─────────────────────
    #   Pull from ultralytics confusion matrix object if available
    cm_raw = None
    if hasattr(results, "confusion_matrix") and results.confusion_matrix is not None:
        cm_raw = results.confusion_matrix.matrix.astype(int)

    # Aggregate into 2×2: row=Actual, col=Predicted
    #   Normal = class 0;  Defective = classes 1-6
    if cm_raw is not None and cm_raw.shape[0] == nc:
        # TN: Actual Normal predicted Normal
        tn = int(cm_raw[0, 0])
        # FP: Actual Normal predicted Defective (any)
        fp = int(cm_raw[0, 1:].sum())
        # FN: Actual Defective predicted Normal
        fn = int(cm_raw[1:, 0].sum())
        # TP: Actual Defective predicted Defective (any correct defect)
        tp = int(cm_raw[1:, 1:].sum())
    else:
        logger.warning("Could not extract raw confusion matrix – using zeros.")
        tp = fp = fn = tn = 0

    binary_metrics = compute_binary_cm(tp, fp, fn, tn)

    # ── Overall map ───────────────────────────────────────────────────────
    map50    = float(results.box.map50)
    map50_95 = float(results.box.map)

    # ── Console report ────────────────────────────────────────────────────
    sep = "─" * 68
    logger.info("\n" + sep)
    logger.info("  COBB 500 CHICK DEFECT DETECTION – EVALUATION REPORT")
    logger.info(sep)
    logger.info(f"  Model      : {model_path}")
    logger.info(f"  Dataset    : {data_yaml}  [test split]")
    logger.info(f"  Conf. thr  : {conf}   |  IoU thr : {iou}")
    logger.info(sep)
    logger.info(f"  mAP@50     : {map50:.4f}")
    logger.info(f"  mAP@50-95  : {map50_95:.4f}")
    logger.info(sep)
    logger.info("  Binary Confusion Matrix  (Normal vs Defective)")
    logger.info(f"    {'':22s}  Pred Normal   Pred Defective")
    logger.info(f"    Actual Normal      :  TN={tn:5d}       FP={fp:5d}")
    logger.info(f"    Actual Defective   :  FN={fn:5d}       TP={tp:5d}")
    logger.info(sep)
    logger.info(f"  Accuracy   : {binary_metrics['accuracy']:.4f}")
    logger.info(f"  Precision  : {binary_metrics['precision']:.4f}")
    logger.info(f"  Recall     : {binary_metrics['recall']:.4f}")
    logger.info(f"  F1-Score   : {binary_metrics['f1']:.4f}")
    logger.info(sep)
    logger.info(f"  {'Class':<22} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AP@50':>10}")
    logger.info(f"  {'─'*22} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")
    for m in metrics_per_class:
        logger.info(
            f"  {m['class_name']:<22} {m['precision']:>10.4f} {m['recall']:>10.4f}"
            f" {m['f1']:>10.4f} {m['ap50']:>10.4f}"
        )
    logger.info(sep + "\n")

    # ── Save CSV report ───────────────────────────────────────────────────
    csv_path = logs_dir / f"eval_report_{ts}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Model", model_path])
        writer.writerow(["mAP@50", f"{map50:.4f}"])
        writer.writerow(["mAP@50-95", f"{map50_95:.4f}"])
        writer.writerow(["Accuracy",  f"{binary_metrics['accuracy']:.4f}"])
        writer.writerow(["Precision", f"{binary_metrics['precision']:.4f}"])
        writer.writerow(["Recall",    f"{binary_metrics['recall']:.4f}"])
        writer.writerow(["F1-Score",  f"{binary_metrics['f1']:.4f}"])
        writer.writerow(["TP", tp])
        writer.writerow(["FP", fp])
        writer.writerow(["FN", fn])
        writer.writerow(["TN", tn])
        writer.writerow([])
        writer.writerow(["Class", "Precision", "Recall", "F1", "AP@50"])
        for m in metrics_per_class:
            writer.writerow([m["class_name"],
                             f"{m['precision']:.4f}",
                             f"{m['recall']:.4f}",
                             f"{m['f1']:.4f}",
                             f"{m['ap50']:.4f}"])
    logger.info(f"CSV report saved → {csv_path}")

    # ── Plots ─────────────────────────────────────────────────────────────
    if cm_raw is not None:
        plot_confusion_matrix(cm_raw, CLASS_NAMES, logs_dir / f"confusion_matrix_{ts}.png")

    plot_per_class_bar(
        [dict(class=m["class_name"], precision=m["precision"],
              recall=m["recall"], f1=m["f1"])
         for m in metrics_per_class],
        logs_dir / f"per_class_metrics_{ts}.png",
    )

    return binary_metrics, metrics_per_class


# ── CLI entry point ───────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Cobb 500 chick defect model")
    p.add_argument("--model",  required=True, help="Path to best.pt")
    p.add_argument("--data",   default=str(ROOT / "data" / "dataset.yaml"))
    p.add_argument("--imgsz",  type=int,   default=640)
    p.add_argument("--conf",   type=float, default=0.40)
    p.add_argument("--iou",    type=float, default=0.45)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args.model, args.data, args.imgsz, args.conf, args.iou, args.device)
