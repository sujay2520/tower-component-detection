"""
train_sample_only.py — Train YOLOv8 for 100 epochs strictly on C:\\Users\\sujay\\Downloads\\sample\\sample.
=======================================================================================================
- Dataset: dataset/data.yaml (124 sample images annotated with LabelImg Pascal VOC & converted to YOLO)
- Epochs: 100
- Device: NVIDIA GeForce GTX 1650 (CUDA)
- Batch Size: 8
"""

import os
import sys
import json
import shutil
import csv
from pathlib import Path
import torch
from ultralytics import YOLO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATASET_YAML = Path("dataset/data.yaml").resolve()
MODEL_BASE = "yolov8n.pt"
EPOCHS = 100
BATCH_SIZE = 8
IMGSZ = 640
PROJECT_DIR = Path("runs").resolve()
RUN_NAME = "tower_sample_100"


def generate_graphs(run_dir, output_dir):
    """Generate high-resolution custom training graphs."""
    run_path = Path(run_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    csv_file = run_path / "results.csv"
    if not csv_file.exists():
        found = list(run_path.rglob("results.csv"))
        if found:
            csv_file = found[0]
            run_path = csv_file.parent
        else:
            print("results.csv not found")
            return

    data = {}
    with open(csv_file, "r") as f:
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

    epochs = data.get("epoch", list(range(len(next(iter(data.values()))))))

    # 1. Losses
    plt.figure(figsize=(10, 6), dpi=200)
    plt.grid(True, linestyle="--", alpha=0.6)
    if "train/box_loss" in data:
        plt.plot(epochs, data["train/box_loss"], label="Train Box Loss", color="#2563eb", lw=2)
    if "val/box_loss" in data:
        plt.plot(epochs, data["val/box_loss"], label="Val Box Loss", color="#3b82f6", lw=2, ls="--")
    if "train/cls_loss" in data:
        plt.plot(epochs, data["train/cls_loss"], label="Train Cls Loss", color="#ea580c", lw=2)
    if "val/cls_loss" in data:
        plt.plot(epochs, data["val/cls_loss"], label="Val Cls Loss", color="#f97316", lw=2, ls="--")
    if "train/dfl_loss" in data:
        plt.plot(epochs, data["train/dfl_loss"], label="Train DFL Loss", color="#16a34a", lw=1.5)
    plt.title("Sample Dataset Training — Losses over 100 Epochs", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path / "training_losses.png")
    plt.close()

    # 2. Metrics
    plt.figure(figsize=(10, 6), dpi=200)
    plt.grid(True, linestyle="--", alpha=0.6)
    if "metrics/precision(B)" in data:
        plt.plot(epochs, data["metrics/precision(B)"], label="Precision", color="#0284c7", lw=2)
    if "metrics/recall(B)" in data:
        plt.plot(epochs, data["metrics/recall(B)"], label="Recall", color="#10b981", lw=2)
    if "metrics/mAP50(B)" in data:
        plt.plot(epochs, data["metrics/mAP50(B)"], label="mAP @ 0.50", color="#7c3aed", lw=2.5)
    if "metrics/mAP50-95(B)" in data:
        plt.plot(epochs, data["metrics/mAP50-95(B)"], label="mAP @ 0.50-0.95", color="#f59e0b", lw=2, ls="--")
    plt.title("Model Accuracy & mAP Progression (Sample Dataset)", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.ylim(0, 1.02)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path / "detection_metrics.png")
    plt.close()

    # 3. 4-Panel Overview
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=200)
    fig.suptitle("Tower Detection Model — 100 Epoch Sample Training Overview", fontsize=16, fontweight="bold")
    for ax in axes.flat:
        ax.grid(True, linestyle="--", alpha=0.5)

    if "train/box_loss" in data:
        axes[0,0].plot(epochs, data["train/box_loss"], label="Train", color="#2563eb", lw=2)
        if "val/box_loss" in data:
            axes[0,0].plot(epochs, data["val/box_loss"], label="Val", color="#3b82f6", ls="--", lw=2)
        axes[0,0].set_title("Box Regression Loss")
        axes[0,0].legend()

    if "train/cls_loss" in data:
        axes[0,1].plot(epochs, data["train/cls_loss"], label="Train", color="#ea580c", lw=2)
        if "val/cls_loss" in data:
            axes[0,1].plot(epochs, data["val/cls_loss"], label="Val", color="#f97316", ls="--", lw=2)
        axes[0,1].set_title("Classification Loss")
        axes[0,1].legend()

    if "metrics/mAP50(B)" in data:
        axes[1,0].plot(epochs, data["metrics/mAP50(B)"], label="mAP50", color="#7c3aed", lw=2.5)
        if "metrics/mAP50-95(B)" in data:
            axes[1,0].plot(epochs, data["metrics/mAP50-95(B)"], label="mAP50-95", color="#f59e0b", ls="--", lw=2)
        axes[1,0].set_title("Mean Average Precision")
        axes[1,0].set_ylim(0, 1.02)
        axes[1,0].legend()

    if "metrics/precision(B)" in data:
        axes[1,1].plot(epochs, data["metrics/precision(B)"], label="Precision", color="#0284c7", lw=2)
        if "metrics/recall(B)" in data:
            axes[1,1].plot(epochs, data["metrics/recall(B)"], label="Recall", color="#10b981", lw=2)
        axes[1,1].set_title("Precision & Recall")
        axes[1,1].set_ylim(0, 1.02)
        axes[1,1].legend()

    plt.tight_layout()
    plt.savefig(out_path / "training_overview.png")
    plt.close()

    # Copy standard YOLO generated plots
    for f in [
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "BoxPR_curve.png",
        "BoxF1_curve.png",
        "results.png",
        "val_batch0_pred.jpg"
    ]:
        src = run_path / f
        if src.exists():
            shutil.copy2(src, out_path / f)


def main():
    print("=" * 65)
    print("  TRAINING YOLOv8 ON SAMPLE/SAMPLE (100 EPOCHS)")
    print("=" * 65)
    print(f"  PyTorch     : {torch.__version__}")
    print(f"  CUDA Device : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"  Base Model  : {MODEL_BASE}")
    print(f"  Dataset YAML: {DATASET_YAML}")
    print(f"  Epochs      : {EPOCHS}")
    print(f"  Batch Size  : {BATCH_SIZE}")
    print(f"  Image Size  : {IMGSZ}")
    print("=" * 65)

    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(MODEL_BASE)

    # Train model
    results = model.train(
        data=str(DATASET_YAML),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH_SIZE,
        patience=30,
        project=str(PROJECT_DIR),
        name=RUN_NAME,
        device=device,
        exist_ok=True,

        # Optimization
        optimizer="AdamW",
        lr0=0.01,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=5,

        # Augmentation
        augment=True,
        mosaic=1.0,
        mixup=0.10,
        flipud=0.5,
        fliplr=0.5,
        degrees=15.0,
        scale=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,

        plots=True,
        save=True,
        save_period=10,
        val=True,
        workers=0,
        seed=42
    )

    best_weights = PROJECT_DIR / RUN_NAME / "weights" / "best.pt"
    if not best_weights.exists():
        candidates = list(PROJECT_DIR.rglob("best.pt"))
        if candidates:
            best_weights = candidates[0]

    print("\n" + "=" * 65)
    print(f"  Training Complete! Evaluating best model: {best_weights}")
    print("=" * 65)

    eval_model = YOLO(str(best_weights))
    metrics = eval_model.val(data=str(DATASET_YAML), split="val", plots=True)

    summary = {
        "dataset": "C:\\Users\\sujay\\Downloads\\sample\\sample (124 images)",
        "model_architecture": "YOLOv8n",
        "epochs_trained": 100,
        "mAP50": round(float(metrics.box.map50), 4),
        "mAP50_95": round(float(metrics.box.map), 4),
        "precision": round(float(metrics.box.mp), 4),
        "recall": round(float(metrics.box.mr), 4),
        "weights": str(best_weights),
        "classes": {}
    }

    for i, name in eval_model.names.items():
        if i < len(metrics.box.p):
            summary["classes"][name] = {
                "precision": round(float(metrics.box.p[i]), 4),
                "recall": round(float(metrics.box.r[i]), 4),
                "mAP50": round(float(metrics.box.ap50[i]), 4),
            }

    print(f"\n  Final mAP@50   : {summary['mAP50']*100:.2f}%")
    print(f"  Final mAP50-95 : {summary['mAP50_95']*100:.2f}%")
    print(f"  Precision      : {summary['precision']*100:.2f}%")
    print(f"  Recall         : {summary['recall']*100:.2f}%")
    for name, cinfo in summary["classes"].items():
        print(f"  [{name}] P={cinfo['precision']:.3f}, R={cinfo['recall']:.3f}, mAP50={cinfo['mAP50']:.3f}")

    # Save metrics JSON in multiple standard locations
    with open("evaluation_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(PROJECT_DIR / RUN_NAME / "evaluation_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Generate custom graphs
    print("\nGenerating training metric graphs...")
    generate_graphs(PROJECT_DIR / RUN_NAME, "training_graphs")

    # Copy best weights to root and standard paths
    shutil.copy2(best_weights, "best_tower_model.pt")
    shutil.copy2(best_weights, "runs/tower_roboflow/weights/best.pt")
    print("Model weights updated: best_tower_model.pt and runs/tower_roboflow/weights/best.pt")

    print("\n" + "=" * 65)
    print("  SAMPLE TRAINING COMPLETE!")
    print("=" * 65)


if __name__ == "__main__":
    main()
