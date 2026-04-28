"""
prepare_dataset.py
==================
Organises raw annotated images into YOLOv11-compatible folder structure
and generates dataset.yaml.

Expected input layout (place your images + YOLO .txt labels here):
  data/images/
    ├── eye_abnormality/
    ├── deformed_beak/
    ├── knock_knees/
    ├── split_legs/
    ├── bend_knees/
    ├── crooked_toes/
    └── normal/

Each folder must contain:
  image.jpg  +  image.txt  (YOLO format: class_id cx cy w h, normalised)

Output structure:
  data/
    images/
      train/ val/ test/
    labels/
      train/ val/ test/
    dataset.yaml

Split ratio: 70% train / 20% val / 10% test  (stratified per class)

Usage:
  conda activate chick_env
  python scripts/prepare_dataset.py --input_dir data/images --output_dir data
"""

import os
import sys
import shutil
import random
import argparse
import logging
import yaml
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Class definitions (must match detector.py CLASS_NAMES) ────────────────
CLASS_MAP = {
    "normal":           0,
    "eye_abnormality":  1,
    "deformed_beak":    2,
    "knock_knees":      3,
    "split_legs":       4,
    "bend_knees":       5,
    "crooked_toes":     6,
}

SPLIT_RATIOS = (0.70, 0.20, 0.10)   # train / val / test

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


# ── helpers ──────────────────────────────────────────────────────────────

def collect_pairs(input_dir: Path) -> dict:
    """
    Walk input_dir and return {class_name: [(img_path, lbl_path), ...]}
    Skips images that have no matching .txt label file.
    """
    pairs = defaultdict(list)
    for class_name in CLASS_MAP:
        class_dir = input_dir / class_name
        if not class_dir.exists():
            logger.warning(f"Class folder not found: {class_dir} – skipping.")
            continue
        for img_path in sorted(class_dir.iterdir()):
            if img_path.suffix.lower() not in IMAGE_EXTS:
                continue
            lbl_path = img_path.with_suffix(".txt")
            if not lbl_path.exists():
                logger.warning(f"No label for {img_path.name} – skipping.")
                continue
            pairs[class_name].append((img_path, lbl_path))
        logger.info(f"  {class_name:20s}: {len(pairs[class_name])} samples")
    return pairs


def stratified_split(pairs: dict, ratios: tuple, seed: int = 42) -> tuple:
    """Return three dicts (train, val, test) with the same keys."""
    random.seed(seed)
    train_all, val_all, test_all = [], [], []
    for class_name, items in pairs.items():
        random.shuffle(items)
        n = len(items)
        n_train = int(n * ratios[0])
        n_val   = int(n * ratios[1])
        train_all.extend(items[:n_train])
        val_all.extend(items[n_train:n_train + n_val])
        test_all.extend(items[n_train + n_val:])
    return train_all, val_all, test_all


def copy_split(items: list, split_name: str, out_dir: Path):
    img_dir = out_dir / "images" / split_name
    lbl_dir = out_dir / "labels" / split_name
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    for img_path, lbl_path in items:
        shutil.copy2(img_path, img_dir / img_path.name)
        shutil.copy2(lbl_path, lbl_dir / lbl_path.name)

    logger.info(f"  {split_name:6s} → {len(items):4d} samples copied")


def write_dataset_yaml(out_dir: Path):
    yaml_path = out_dir / "dataset.yaml"
    config = {
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val":   "images/val",
        "test":  "images/test",
        "nc":    len(CLASS_MAP),
        "names": {v: k.replace("_", " ").title() for k, v in CLASS_MAP.items()},
    }
    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    logger.info(f"dataset.yaml written → {yaml_path}")
    return yaml_path


def augment_minority_class(items: list, target_count: int, out_dir: Path,
                           split_name: str = "train"):
    """
    Simple duplication-based augmentation for the minority class (Normal chicks).
    More sophisticated augmentation is handled by YOLOv11's built-in pipeline during training.
    This ensures the folder contains enough files so YOLO doesn't complain.
    """
    import cv2, numpy as np
    img_dir = out_dir / "images" / split_name
    lbl_dir = out_dir / "labels" / split_name

    current = len(items)
    added = 0
    idx = 0
    while current + added < target_count:
        img_path, lbl_path = items[idx % current]
        img = cv2.imread(str(img_path))
        if img is None:
            idx += 1
            continue

        # Apply a random horizontal flip
        aug_img = cv2.flip(img, 1)
        suffix = f"_aug{added:04d}"
        new_img_path = img_dir / (img_path.stem + suffix + img_path.suffix)
        new_lbl_path = lbl_dir / (lbl_path.stem + suffix + ".txt")

        cv2.imwrite(str(new_img_path), aug_img)

        # Flip bounding boxes: new_cx = 1 - cx (YOLO format, normalised)
        lines = lbl_path.read_text().strip().splitlines()
        new_lines = []
        for line in lines:
            parts = line.split()
            if len(parts) == 5:
                cls, cx, cy, w, h = parts
                new_lines.append(f"{cls} {1.0 - float(cx):.6f} {cy} {w} {h}")
            else:
                new_lines.append(line)
        new_lbl_path.write_text("\n".join(new_lines))

        added += 1
        idx += 1

    logger.info(f"  Augmented minority class: added {added} flipped copies")


# ── main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Prepare Cobb 500 chick dataset")
    parser.add_argument("--input_dir",  default=str(ROOT / "data" / "images"),
                        help="Root folder containing per-class subfolders")
    parser.add_argument("--output_dir", default=str(ROOT / "data"),
                        help="Output directory for the processed dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no_augment", action="store_true",
                        help="Skip minority-class augmentation")
    args = parser.parse_args()

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    logger.info("=== Cobb 500 Dataset Preparation ===")
    logger.info(f"Input : {input_dir}")
    logger.info(f"Output: {output_dir}")

    # 1. Collect image–label pairs
    logger.info("Scanning class folders …")
    pairs = collect_pairs(input_dir)
    if not pairs:
        logger.error("No valid image-label pairs found. Exiting.")
        sys.exit(1)

    total = sum(len(v) for v in pairs.values())
    logger.info(f"Total valid samples: {total}")

    # 2. Stratified split
    logger.info("Splitting dataset …")
    train_items, val_items, test_items = stratified_split(pairs, SPLIT_RATIOS, args.seed)

    # 3. Copy to output folders
    logger.info("Copying files …")
    copy_split(train_items, "train", output_dir)
    copy_split(val_items,   "val",   output_dir)
    copy_split(test_items,  "test",  output_dir)

    # 4. Augment minority class (Normal chicks = 10 samples)
    if not args.no_augment:
        normal_train = [(i, l) for i, l in train_items
                        if i.parent.name == "normal"]
        if normal_train:
            # Target: at least as many Normal samples as the smallest defect class
            defect_counts = [len(v) for k, v in pairs.items() if k != "normal"]
            target = min(defect_counts) if defect_counts else len(normal_train)
            target_train = int(target * SPLIT_RATIOS[0])
            if len(normal_train) < target_train:
                logger.info("Augmenting minority (normal) class …")
                augment_minority_class(normal_train, target_train, output_dir, "train")

    # 5. Write dataset.yaml
    write_dataset_yaml(output_dir)

    logger.info("\n✓ Dataset preparation complete.")
    logger.info(f"  Train : {len(train_items)} samples")
    logger.info(f"  Val   : {len(val_items)} samples")
    logger.info(f"  Test  : {len(test_items)} samples")


if __name__ == "__main__":
    main()
