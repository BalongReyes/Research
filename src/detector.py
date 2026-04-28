"""
detector.py
===========
Core YOLOv11-based defect detection engine for Cobb 500 day-old chicks.

Preprocessing pipeline:
  - Resize to model input size
  - CLAHE (Contrast Limited Adaptive Histogram Equalization) for lighting normalization
  - Gaussian blur for noise reduction
  - Direct YOLOv11 inference (NO Canny edge detection)

Defect classes:
  D1 - Eye Abnormality
  D2 - Deformed Beak
  D3 - Knock Knees
  D4 - Split Legs
  D5 - Bend Knees
  D6 - Crooked Toes
  D0 - Normal (healthy chick)
"""

import cv2
import numpy as np
import time
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None  # Allows import without ultralytics for unit tests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLASS_NAMES = {
    0: "Normal",
    1: "Eye_Abnormality",
    2: "Deformed_Beak",
    3: "Knock_Knees",
    4: "Split_Legs",
    5: "Bend_Knees",
    6: "Crooked_Toes",
}

DEFECT_CLASSES = {k for k, v in CLASS_NAMES.items() if v != "Normal"}

# Temporal logic thresholds (seconds)
CONFIRM_THRESHOLD   = 5.0   # Defect must persist this long to be confirmed
GRACE_PERIOD        = 2.0   # Gap allowed before resetting persistence counter
INSPECTION_WINDOW   = 15.0  # Total inspection time per chick

# CLAHE parameters
CLAHE_CLIP_LIMIT    = 2.0
CLAHE_TILE_GRID     = (8, 8)

# Confidence threshold for detections
CONF_THRESHOLD      = 0.40
IOU_THRESHOLD       = 0.45


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple   # (x1, y1, x2, y2) in pixel coords

@dataclass
class TemporalState:
    """Tracks persistence of each defect class over time."""
    persistence: dict = field(default_factory=lambda: {k: 0.0 for k in DEFECT_CLASSES})
    grace:       dict = field(default_factory=lambda: {k: 0.0 for k in DEFECT_CLASSES})
    confirmed:   set  = field(default_factory=set)
    total_elapsed: float = 0.0
    chick_present: bool  = False

    def reset(self):
        self.persistence = {k: 0.0 for k in DEFECT_CLASSES}
        self.grace       = {k: 0.0 for k in DEFECT_CLASSES}
        self.confirmed   = set()
        self.total_elapsed = 0.0
        self.chick_present = False


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess_frame(frame: np.ndarray, target_size: tuple = (640, 640)) -> np.ndarray:
    """
    Prepare a BGR camera frame for YOLOv11 inference.

    Steps
    -----
    1. Resize with letterboxing to preserve aspect ratio.
    2. Convert to LAB color space.
    3. Apply CLAHE on the L channel to normalize illumination variation.
    4. Reconstruct BGR frame and apply mild Gaussian smoothing.

    Note: Canny edge detection is intentionally NOT used. CLAHE provides
    sufficient contrast enhancement for YOLOv11 to extract its own learned
    features without artificially destroying color or texture information.
    """
    # 1. Letterbox resize
    h, w = frame.shape[:2]
    scale = min(target_size[0] / h, target_size[1] / w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Pad to square
    pad_h = target_size[0] - new_h
    pad_w = target_size[1] - new_w
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))

    # 2-3. CLAHE on L channel
    lab = cv2.cvtColor(padded, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    enhanced = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    # 4. Mild Gaussian denoise (3×3 kernel – preserves detail)
    processed = cv2.GaussianBlur(enhanced, (3, 3), 0)

    return processed


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class ChickDefectDetector:
    """
    Wraps a YOLOv11 model and applies temporal validation logic
    to produce a final chick classification result.
    """

    def __init__(self, model_path: str, device: str = "cuda"):
        if YOLO is None:
            raise RuntimeError("ultralytics is not installed. Run: pip install ultralytics")

        self.model_path = Path(model_path)
        self.device = device
        self.model: Optional[YOLO] = None
        self.state = TemporalState()
        self._last_ts: Optional[float] = None

        logger.info(f"Loading model from {model_path} on device={device}")
        self._load_model()

    def _load_model(self):
        self.model = YOLO(str(self.model_path))
        self.model.to(self.device)
        logger.info("Model loaded successfully.")

    # ------------------------------------------------------------------
    # Per-frame inference
    # ------------------------------------------------------------------

    def infer(self, frame: np.ndarray) -> list[Detection]:
        """
        Run inference on a raw BGR frame.
        Returns a list of Detection objects.
        """
        processed = preprocess_frame(frame)

        results = self.model.predict(
            source=processed,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            verbose=False,
            device=self.device,
        )

        detections = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0].item())
                conf   = float(box.conf[0].item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append(Detection(
                    class_id=cls_id,
                    class_name=CLASS_NAMES.get(cls_id, "Unknown"),
                    confidence=conf,
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                ))

        return detections

    # ------------------------------------------------------------------
    # Temporal validation
    # ------------------------------------------------------------------

    def update_temporal_state(self, detections: list[Detection]) -> TemporalState:
        """
        Update persistence / grace counters based on current detections.
        Must be called once per frame, in real time.
        """
        now = time.monotonic()
        if self._last_ts is None:
            self._last_ts = now
        dt = now - self._last_ts
        self._last_ts = now

        detected_classes = {d.class_id for d in detections}

        # Check if any chick is in the frame (class 0 = Normal counts too)
        self.state.chick_present = bool(detected_classes)

        if not self.state.chick_present:
            # No chick detected – reset everything
            self.state.reset()
            self._last_ts = None
            return self.state

        self.state.total_elapsed += dt

        for cls_id in DEFECT_CLASSES:
            if cls_id in detected_classes:
                # Defect visible this frame
                self.state.persistence[cls_id] += dt
                self.state.grace[cls_id] = 0.0
            else:
                # Defect not visible – accumulate grace period
                self.state.grace[cls_id] += dt
                if self.state.grace[cls_id] > GRACE_PERIOD:
                    # Reset persistence if gap is too long
                    self.state.persistence[cls_id] = 0.0
                    self.state.grace[cls_id] = 0.0

            # Confirm if defect persisted long enough
            if self.state.persistence[cls_id] >= CONFIRM_THRESHOLD:
                self.state.confirmed.add(cls_id)

        return self.state

    # ------------------------------------------------------------------
    # Final verdict
    # ------------------------------------------------------------------

    def get_verdict(self) -> dict:
        """
        After the inspection window, return the final classification.
        """
        is_defective = bool(self.state.confirmed)
        confirmed_names = [CLASS_NAMES[c] for c in self.state.confirmed]
        return {
            "result":    "Defective" if is_defective else "Normal",
            "defects":   confirmed_names,
            "elapsed_s": round(self.state.total_elapsed, 2),
        }

    def reset(self):
        self.state.reset()
        self._last_ts = None
