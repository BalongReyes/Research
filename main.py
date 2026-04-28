"""
main.py
=======
Entry point for the Computer Vision–Based Defective Cobb 500 Chick
Identification System running on the Jetson Orin Nano 8GB Developer Kit.

Hardware
--------
  Processing : Jetson Orin Nano 8GB (128GB SD card)
  Cameras    : 2× A4TECH 1080p USB Webcam (indices 0 and 2)
  Display    : 7-inch LCD (1024×600)
  Servo      : Tower Pro – GPIO pin 32
  Lighting   : 12V LED strip (always-on, no software control needed)

Usage
-----
  # Activate environment first
  conda activate chick_env

  # Run with trained model
  python main.py --model models/<run>/weights/best.pt

  # Simulation mode (no physical hardware required – for development)
  python main.py --model models/<run>/weights/best.pt --simulate

  # Optional flags
  --cam0      int    Camera index for left  camera  (default: 0)
  --cam1      int    Camera index for right camera  (default: 2)
  --device    str    cuda | cpu              (default: cuda)
  --conf      float  Detection confidence   (default: 0.40)
  --log_level str    DEBUG | INFO | WARNING (default: INFO)
"""

import sys
import argparse
import logging
from pathlib import Path

# ── Ensure project root is importable ────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from utils.logger         import setup_logging
from src.detector         import ChickDefectDetector, CONF_THRESHOLD, IOU_THRESHOLD
from src.camera_manager   import DualCameraManager
from src.servo_controller import ServoController
from gui.gui              import launch_gui


# ── Argument parsing ─────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Cobb 500 Chick Defect Detection System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model",     required=True,
                   help="Path to trained YOLOv11 weights (best.pt)")
    p.add_argument("--cam0",      type=int,   default=0,
                   help="Camera index for left/front camera")
    p.add_argument("--cam1",      type=int,   default=2,
                   help="Camera index for right/side camera")
    p.add_argument("--device",    default="cuda",
                   help="Inference device: 'cuda' or 'cpu'")
    p.add_argument("--conf",      type=float, default=CONF_THRESHOLD,
                   help="YOLOv11 detection confidence threshold")
    p.add_argument("--iou",       type=float, default=IOU_THRESHOLD,
                   help="NMS IoU threshold")
    p.add_argument("--simulate",  action="store_true",
                   help="Enable simulation mode (no GPIO / servo hardware)")
    p.add_argument("--log_level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument("--no_gui",    action="store_true",
                   help="Headless mode: print verdicts to console only")
    return p.parse_args()


# ── Startup checks ───────────────────────────────────────────────────────

def check_model(model_path: str) -> Path:
    p = Path(model_path)
    if not p.exists():
        raise FileNotFoundError(
            f"Model weights not found: {p}\n"
            "Train first with:  python scripts/train.py --model yolo11n.pt"
        )
    return p


# ── Headless loop (no GUI) ───────────────────────────────────────────────

def run_headless(detector, cam_manager, servo_ctrl):
    """
    Continuous headless inspection loop.
    Each chick is inspected for up to 15 seconds.
    Press Ctrl+C to exit.
    """
    import time
    import cv2

    logger = logging.getLogger(__name__)
    logger.info("Running in HEADLESS mode. Press Ctrl+C to stop.")

    chick_count = 0

    try:
        while True:
            detector.reset()
            logger.info(f"--- Waiting for chick #{chick_count + 1} ---")

            while True:
                f0, f1 = cam_manager.read_pair()
                if f0 is None:
                    time.sleep(0.05)
                    continue

                detections = detector.infer(f0)
                state      = detector.update_temporal_state(detections)

                if not state.chick_present:
                    time.sleep(0.05)
                    continue

                # Chick detected – run until inspection window closes
                while state.total_elapsed < 15.0:
                    f0, _ = cam_manager.read_pair()
                    if f0 is None:
                        time.sleep(0.03)
                        continue
                    detections = detector.infer(f0)
                    state      = detector.update_temporal_state(detections)
                    time.sleep(0.033)

                # Verdict
                verdict = detector.get_verdict()
                chick_count += 1
                logger.info(
                    f"Chick #{chick_count} → {verdict['result']} "
                    f"| Defects: {', '.join(verdict['defects']) or 'None'} "
                    f"| Duration: {verdict['elapsed_s']}s"
                )

                if verdict["result"] == "Normal":
                    servo_ctrl.open_healthy_hatch()
                else:
                    servo_ctrl.open_defect_hatch()

                break   # back to outer loop for next chick

    except KeyboardInterrupt:
        logger.info(f"Stopped. Total chicks inspected: {chick_count}")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Logging
    setup_logging(level=args.log_level)
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("  Cobb 500 Chick Defect Detection System  –  Starting")
    logger.info("=" * 60)

    # Model
    model_path = check_model(args.model)
    logger.info(f"Model  : {model_path}")
    logger.info(f"Device : {args.device}")
    logger.info(f"Conf   : {args.conf}   IoU: {args.iou}")

    # Detector
    detector = ChickDefectDetector(
        model_path=str(model_path),
        device=args.device,
    )
    # Override thresholds from CLI
    detector.model.overrides["conf"] = args.conf
    detector.model.overrides["iou"]  = args.iou

    # Cameras
    cam_manager = DualCameraManager(
        cam0_index=args.cam0,
        cam1_index=args.cam1,
    )
    if not cam_manager.start():
        logger.error("Failed to open cameras. Check USB connections and indices.")
        sys.exit(1)
    logger.info(f"Cameras : /dev/video{args.cam0} (left)  /dev/video{args.cam1} (right)")

    # Servo
    servo_ctrl = ServoController(simulate=args.simulate)
    if args.simulate:
        logger.info("Servo  : SIMULATION MODE (no GPIO output)")
    else:
        logger.info("Servo  : GPIO pin 32 (hardware mode)")

    # Launch
    try:
        if args.no_gui:
            run_headless(detector, cam_manager, servo_ctrl)
        else:
            launch_gui(detector, cam_manager, servo_ctrl)
    finally:
        cam_manager.stop()
        servo_ctrl.cleanup()
        logger.info("System shutdown complete.")


if __name__ == "__main__":
    main()
