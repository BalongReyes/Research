"""
gui.py
======
PyQt5 Graphical User Interface for the Computer Vision–Based
Defective Cobb 500 Chick Identification System.

Displays:
  - Live dual-camera feed (side-by-side)
  - Bounding boxes with defect labels
  - Real-time detection log
  - Classification verdict (Normal / Defective)
  - Temporal progress bar (inspection timer)
  - System status and servo action log

Compatible with the 7-inch LCD display connected to the Jetson Orin Nano.
Resolution target: 1024×600 (common 7" panel)
"""

import sys
import time
import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Optional

from utils.session_logger import SessionLogger

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout,
    QHBoxLayout, QPushButton, QTextEdit, QProgressBar, QFrame, QGridLayout,
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QMutex
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor, QPalette

logger = logging.getLogger(__name__)

# Color palette
COLOR_NORMAL  = "#27AE60"   # green
COLOR_DEFECT  = "#E74C3C"   # red
COLOR_PENDING = "#F39C12"   # amber
COLOR_BG      = "#1A1A2E"   # dark navy
COLOR_PANEL   = "#16213E"   # panel bg
COLOR_TEXT    = "#EAEAEA"   # light text
COLOR_ACCENT  = "#0F3460"   # accent


def bgr_to_qimage(frame: np.ndarray) -> QImage:
    """Convert an OpenCV BGR frame to a QImage."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    return QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)


def draw_detections(frame: np.ndarray, detections: list, confirmed: set) -> np.ndarray:
    """Draw bounding boxes and labels on a copy of the frame."""
    vis = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        is_confirmed = det.class_id in confirmed
        is_defect    = det.class_name != "Normal"

        # Color: red if defect confirmed, orange if pending, green if normal
        if not is_defect:
            color = (39, 174, 96)
        elif is_confirmed:
            color = (231, 76, 60)
        else:
            color = (243, 156, 18)

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"{det.class_name} {det.confidence:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(vis, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return vis


# ---------------------------------------------------------------------------
# Worker thread – runs detection loop without blocking the GUI
# ---------------------------------------------------------------------------

class DetectionWorker(QThread):
    frame_ready    = pyqtSignal(np.ndarray, list)
    state_updated  = pyqtSignal(object)
    verdict_ready  = pyqtSignal(dict)
    log_message    = pyqtSignal(str)

    def __init__(self, detector, cam_manager, session_log, parent=None):
        super().__init__(parent)
        self.detector    = detector
        self.cam_manager = cam_manager
        self.session_log = session_log
        self._running    = False
        self._mutex      = QMutex()

    def run(self):
        self._running = True
        self.detector.reset()
        max_conf = 0.0

        while self._running:
            frame_left, frame_right = self.cam_manager.read_pair()

            if frame_left is None:
                time.sleep(0.03)
                continue

            detections = self.detector.infer(frame_left)
            state      = self.detector.update_temporal_state(detections)

            # Track max confidence
            for d in detections:
                if d.confidence > max_conf:
                    max_conf = d.confidence

            # Annotate left frame; embed right frame as inset
            annotated = draw_detections(frame_left, detections, state.confirmed)
            if frame_right is not None:
                inset = cv2.resize(frame_right, (240, 135))
                annotated[10:145, annotated.shape[1]-250:annotated.shape[1]-10] = inset
                cv2.rectangle(annotated,
                              (annotated.shape[1]-250, 10),
                              (annotated.shape[1]-10, 145),
                              (255, 255, 255), 1)

            self.frame_ready.emit(annotated, detections)
            self.state_updated.emit(state)

            if state.total_elapsed >= 15.0:
                verdict = self.detector.get_verdict()
                # Record to CSV
                chick_num = self.session_log.record(verdict, confidence_max=max_conf)
                verdict["chick_num"]   = chick_num
                verdict["max_conf"]    = round(max_conf, 2)
                self.verdict_ready.emit(verdict)
                self.log_message.emit(
                    f"Chick #{chick_num} → {verdict['result']} | "
                    f"Defects: {', '.join(verdict['defects']) or 'None'} | "
                    f"Conf: {max_conf:.0%}"
                )
                self._running = False
                break

            time.sleep(0.033)

    def stop(self):
        self._running = False
        self.wait()


# ---------------------------------------------------------------------------
# Main Window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):

    def __init__(self, detector, cam_manager, servo_ctrl):
        super().__init__()
        self.detector    = detector
        self.cam_manager = cam_manager
        self.servo_ctrl  = servo_ctrl
        self.worker: Optional[DetectionWorker] = None

        self.setWindowTitle("Cobb 500 Chick Defect Detection System")
        self.setStyleSheet(f"background-color: {COLOR_BG}; color: {COLOR_TEXT};")
        self.showMaximized()

        self._build_ui()

    # ------------------------------------------------------------------
    # UI layout
    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        main_layout = QHBoxLayout(root)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # --- Left column: camera feed ---
        left = QVBoxLayout()

        title = QLabel("Cobb 500 Chick Defect Detection System")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(f"color: {COLOR_TEXT}; padding: 6px;")
        left.addWidget(title)

        self.cam_label = QLabel("Waiting for camera…")
        self.cam_label.setAlignment(Qt.AlignCenter)
        self.cam_label.setMinimumSize(640, 480)
        self.cam_label.setStyleSheet(
            f"background-color: #000; border: 2px solid {COLOR_ACCENT}; border-radius: 4px;"
        )
        left.addWidget(self.cam_label, stretch=1)

        # Inspection timer bar
        timer_row = QHBoxLayout()
        timer_lbl = QLabel("Inspection:")
        timer_lbl.setFont(QFont("Arial", 10))
        self.timer_bar = QProgressBar()
        self.timer_bar.setRange(0, 150)   # 0–15 s × 10
        self.timer_bar.setValue(0)
        self.timer_bar.setTextVisible(True)
        self.timer_bar.setFormat("%v / 150 (×0.1 s)")
        self.timer_bar.setStyleSheet(
            "QProgressBar { border: 1px solid #444; border-radius: 4px; background: #0a0a0a; }"
            f"QProgressBar::chunk {{ background: {COLOR_ACCENT}; }}"
        )
        timer_row.addWidget(timer_lbl)
        timer_row.addWidget(self.timer_bar)
        left.addLayout(timer_row)

        main_layout.addLayout(left, stretch=3)

        # --- Right column: verdict + log ---
        right = QVBoxLayout()
        right.setSpacing(8)

        # Verdict panel
        self.verdict_frame = QFrame()
        self.verdict_frame.setStyleSheet(
            f"background-color: {COLOR_PANEL}; border: 2px solid {COLOR_ACCENT}; border-radius: 6px;"
        )
        vf_layout = QVBoxLayout(self.verdict_frame)

        self.verdict_label = QLabel("PENDING")
        self.verdict_label.setFont(QFont("Arial", 28, QFont.Bold))
        self.verdict_label.setAlignment(Qt.AlignCenter)
        self.verdict_label.setStyleSheet(f"color: {COLOR_PENDING};")
        vf_layout.addWidget(self.verdict_label)

        self.defect_list_label = QLabel("")
        self.defect_list_label.setAlignment(Qt.AlignCenter)
        self.defect_list_label.setWordWrap(True)
        self.defect_list_label.setFont(QFont("Arial", 11))
        vf_layout.addWidget(self.defect_list_label)

        right.addWidget(self.verdict_frame)

        # Detection stats grid
        stats_frame = QFrame()
        stats_frame.setStyleSheet(
            f"background-color: {COLOR_PANEL}; border: 1px solid {COLOR_ACCENT}; border-radius: 4px;"
        )
        grid = QGridLayout(stats_frame)
        grid.setContentsMargins(8, 8, 8, 8)
        grid.setSpacing(4)

        self._defect_indicators = {}
        defect_names = [
            "Eye Abnormality", "Deformed Beak", "Knock Knees",
            "Split Legs", "Bend Knees", "Crooked Toes"
        ]
        for i, name in enumerate(defect_names):
            lbl = QLabel(name)
            lbl.setFont(QFont("Arial", 9))
            ind = QLabel("○")
            ind.setFont(QFont("Arial", 14))
            ind.setAlignment(Qt.AlignCenter)
            ind.setStyleSheet("color: #666;")
            grid.addWidget(lbl, i, 0)
            grid.addWidget(ind, i, 1)
            self._defect_indicators[i+1] = ind   # class IDs 1-6

        right.addWidget(stats_frame)

        # Log
        log_lbl = QLabel("Detection Log")
        log_lbl.setFont(QFont("Arial", 10, QFont.Bold))
        right.addWidget(log_lbl)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setFont(QFont("Courier New", 9))
        self.log_box.setStyleSheet(
            f"background-color: #0a0a0a; color: {COLOR_TEXT}; border: 1px solid {COLOR_ACCENT};"
        )
        right.addWidget(self.log_box, stretch=1)

        # Controls
        btn_row = QHBoxLayout()
        self.start_btn = QPushButton("▶  Start Inspection")
        self.start_btn.setFont(QFont("Arial", 12, QFont.Bold))
        self.start_btn.setStyleSheet(
            f"background-color: {COLOR_NORMAL}; color: white; padding: 10px; border-radius: 4px;"
        )
        self.start_btn.clicked.connect(self._start_inspection)

        self.reset_btn = QPushButton("↺  Reset")
        self.reset_btn.setFont(QFont("Arial", 12))
        self.reset_btn.setStyleSheet(
            f"background-color: {COLOR_ACCENT}; color: white; padding: 10px; border-radius: 4px;"
        )
        self.reset_btn.clicked.connect(self._reset)

        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.reset_btn)
        right.addLayout(btn_row)

        main_layout.addLayout(right, stretch=1)

    # ------------------------------------------------------------------
    # Inspection control
    # ------------------------------------------------------------------

    def _start_inspection(self):
        self._reset_ui()
        
        # Instantiate the SessionLogger
        session_log = SessionLogger()
        
        # Pass session_log to the DetectionWorker
        self.worker = DetectionWorker(self.detector, self.cam_manager, session_log)
        
        self.worker.frame_ready.connect(self._on_frame)
        self.worker.state_updated.connect(self._on_state)
        self.worker.verdict_ready.connect(self._on_verdict)
        self.worker.log_message.connect(self._log)
        self.worker.start()
        self.start_btn.setEnabled(False)
        self._log("Inspection started.")

    def _reset(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
        self.detector.reset()
        self._reset_ui()
        self.start_btn.setEnabled(True)
        self._log("System reset.")

    def _reset_ui(self):
        self.verdict_label.setText("PENDING")
        self.verdict_label.setStyleSheet(f"color: {COLOR_PENDING};")
        self.defect_list_label.setText("")
        self.timer_bar.setValue(0)
        for ind in self._defect_indicators.values():
            ind.setText("○")
            ind.setStyleSheet("color: #666;")

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_frame(self, frame: np.ndarray, detections: list):
        qimg = bgr_to_qimage(frame)
        pix  = QPixmap.fromImage(qimg)
        self.cam_label.setPixmap(
            pix.scaled(self.cam_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def _on_state(self, state):
        # Update timer bar
        val = min(int(state.total_elapsed * 10), 150)
        self.timer_bar.setValue(val)

        # Update defect indicators
        for cls_id, ind in self._defect_indicators.items():
            if cls_id in state.confirmed:
                ind.setText("●")
                ind.setStyleSheet(f"color: {COLOR_DEFECT};")
            elif state.persistence.get(cls_id, 0) > 0:
                ind.setText("◑")
                ind.setStyleSheet(f"color: {COLOR_PENDING};")

    def _on_verdict(self, verdict: dict):
        if verdict["result"] == "Normal":
            self.verdict_label.setText("✓  NORMAL")
            self.verdict_label.setStyleSheet(f"color: {COLOR_NORMAL}; font-size: 28px;")
            self.defect_list_label.setText("Class A – Healthy Chick")
            self._trigger_servo("normal")
        else:
            self.verdict_label.setText("✗  DEFECTIVE")
            self.verdict_label.setStyleSheet(f"color: {COLOR_DEFECT}; font-size: 28px;")
            self.defect_list_label.setText("Defects: " + ", ".join(verdict["defects"]))
            self._trigger_servo("defective")

        self.timer_bar.setValue(150)
        self.start_btn.setEnabled(True)

    def _trigger_servo(self, verdict_type: str):
        """Run servo in a background thread to avoid GUI freeze."""
        import threading
        if verdict_type == "normal":
            t = threading.Thread(target=self.servo_ctrl.open_healthy_hatch, daemon=True)
        else:
            t = threading.Thread(target=self.servo_ctrl.open_defect_hatch, daemon=True)
        t.start()

    def _log(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        self.log_box.append(f"[{ts}] {msg}")

    def closeEvent(self, event):
        self._reset()
        self.servo_ctrl.cleanup()
        self.cam_manager.stop()
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def launch_gui(detector, cam_manager, servo_ctrl):
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow(detector, cam_manager, servo_ctrl)
    window.show()
    sys.exit(app.exec_())
