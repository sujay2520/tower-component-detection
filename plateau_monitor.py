"""
plateau_monitor.py — Monitors training results.csv for metric plateaus.
Checks if mAP50, val_cls_loss, or val_box_loss have flattened into a
horizontal line over the last N epochs. If plateau detected, kills training
and reports to the user.
"""

import csv
import sys
import time
import signal
import os
from pathlib import Path
import numpy as np

RESULTS_CSV = Path("runs/tower_aug_86/results.csv")
PLATEAU_WINDOW = 8          # Number of consecutive epochs to check
MAP_TOLERANCE = 0.005       # mAP50 change < 0.5% over window = plateau
LOSS_TOLERANCE = 0.008      # loss change < 0.8% over window = plateau
CHECK_INTERVAL = 60         # Check every 60 seconds


def read_results():
    """Read results.csv and return parsed data."""
    if not RESULTS_CSV.exists():
        return None
    data = {}
    with open(RESULTS_CSV, "r") as f:
        reader = csv.reader(f)
        header = [c.strip() for c in next(reader)]
        for col in header:
            data[col] = []
        for row in reader:
            for col, val in zip(header, row):
                try:
                    data[col].append(float(val.strip()))
                except ValueError:
                    data[col].append(0.0)
    return data


def check_plateau(values, tolerance):
    """Check if the last PLATEAU_WINDOW values form a flat line."""
    if len(values) < PLATEAU_WINDOW:
        return False, 0.0

    window = values[-PLATEAU_WINDOW:]
    val_range = max(window) - min(window)
    std_dev = float(np.std(window))

    # Plateau = range AND std dev are both tiny
    is_flat = val_range < tolerance and std_dev < tolerance * 0.6
    return is_flat, val_range


def analyze():
    """Analyze results.csv for plateaus."""
    data = read_results()
    if data is None:
        return None, "results.csv not found"

    epoch_count = len(data.get("epoch", []))
    if epoch_count < PLATEAU_WINDOW + 3:
        return None, f"Only {epoch_count} epochs completed, need at least {PLATEAU_WINDOW + 3}"

    current_epoch = int(data["epoch"][-1]) if data.get("epoch") else 0

    plateau_signals = []

    # Check mAP50
    if "metrics/mAP50(B)" in data:
        is_flat, val_range = check_plateau(data["metrics/mAP50(B)"], MAP_TOLERANCE)
        if is_flat:
            plateau_signals.append(f"mAP50 FLAT (range={val_range:.4f} over last {PLATEAU_WINDOW} epochs)")

    # Check mAP50-95
    if "metrics/mAP50-95(B)" in data:
        is_flat, val_range = check_plateau(data["metrics/mAP50-95(B)"], MAP_TOLERANCE)
        if is_flat:
            plateau_signals.append(f"mAP50-95 FLAT (range={val_range:.4f} over last {PLATEAU_WINDOW} epochs)")

    # Check val box loss
    if "val/box_loss" in data:
        is_flat, val_range = check_plateau(data["val/box_loss"], LOSS_TOLERANCE)
        if is_flat:
            plateau_signals.append(f"Val Box Loss FLAT (range={val_range:.4f} over last {PLATEAU_WINDOW} epochs)")

    # Check val cls loss
    if "val/cls_loss" in data:
        is_flat, val_range = check_plateau(data["val/cls_loss"], LOSS_TOLERANCE)
        if is_flat:
            plateau_signals.append(f"Val Cls Loss FLAT (range={val_range:.4f} over last {PLATEAU_WINDOW} epochs)")

    # Need at least 2 plateau signals (mAP + any loss) to be sure
    if len(plateau_signals) >= 2:
        return True, f"PLATEAU DETECTED at epoch {current_epoch}:\n  " + "\n  ".join(plateau_signals)

    # Report current status
    map50 = data["metrics/mAP50(B)"][-1] if "metrics/mAP50(B)" in data else 0
    map95 = data["metrics/mAP50-95(B)"][-1] if "metrics/mAP50-95(B)" in data else 0
    val_cls = data["val/cls_loss"][-1] if "val/cls_loss" in data else 0
    return False, f"Epoch {current_epoch}/86 — mAP50={map50:.4f}, mAP50-95={map95:.4f}, ValClsLoss={val_cls:.4f} — NO PLATEAU (checked {len(plateau_signals)} flat signals)"


def main():
    print("=" * 60)
    print("  PLATEAU MONITOR — Watching training for flat-line metrics")
    print("=" * 60)
    print(f"  Window: {PLATEAU_WINDOW} epochs | mAP tolerance: {MAP_TOLERANCE}")
    print(f"  Loss tolerance: {LOSS_TOLERANCE} | Check interval: {CHECK_INTERVAL}s")
    print("=" * 60)

    while True:
        is_plateau, message = analyze()

        if is_plateau is None:
            print(f"[WAIT] {message}")
        elif is_plateau:
            print(f"\n{'!'*60}")
            print(f"  ⚠️  {message}")
            print(f"{'!'*60}")
            print("  ACTION: Training should be stopped — metrics have plateaued.")
            # Write signal file for external monitoring
            Path("PLATEAU_DETECTED.flag").write_text(message)
            sys.exit(0)
        else:
            print(f"[OK] {message}")

        # Check if training is still running
        if RESULTS_CSV.exists():
            data = read_results()
            if data and data.get("epoch"):
                current = int(data["epoch"][-1])
                if current >= 86:
                    print(f"\n[DONE] Training completed all 86 epochs!")
                    sys.exit(0)

        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()
