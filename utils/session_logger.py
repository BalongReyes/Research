import csv
import time
from pathlib import Path

class SessionLogger:
    def __init__(self, log_dir="logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.chick_count = 0
        
        # Create a new CSV file for this session
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.csv_path = self.log_dir / f"session_{ts}.csv"
        
        with open(self.csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["chick_num", "result", "defects", "confidence_max", "elapsed_s", "timestamp"])

    def record(self, verdict: dict, confidence_max: float) -> int:
        """Logs the chick inspection results and returns the chick number."""
        self.chick_count += 1
        with open(self.csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                self.chick_count,
                verdict.get("result", "Unknown"),
                ", ".join(verdict.get("defects", [])),
                round(confidence_max, 2),
                verdict.get("elapsed_s", 0.0),
                time.strftime("%H:%M:%S")
            ])
        return self.chick_count
