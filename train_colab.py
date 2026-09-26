"""
train_colab.py -- Google Colab Training Script for Tower Component Detection
=============================================================================
YOLOv8n trained on merged dataset (Roboflow + Supporting + Mono + Sample)
Classes: ['monopole_tower', 'supporting_tower']

INSTRUCTIONS:
1. Upload this script AND the entire dataset_merged/ folder to Google Drive
2. In Colab, mount Google Drive and run this script
3. After training, download the best.pt weights file

Or run locally with:
    python train_colab.py
"""

import os
import sys
import json
from pathlib import Path

# ---- Auto-install dependencies (works in Colab and local) ----
def install_deps():
    try:
        import ultralytics
    except ImportError:
        os.system("pip install ultralytics>=8.2.0")
    try:
        import matplotlib
    except ImportError:
        os.system("pip install matplotlib")

install_deps()

import torch
from ultralytics import YOLO

# =====================================================
#  CONFIGURATION — Edit these for your environment
# =====================================================

# If running in Colab with Drive mounted:
COLAB_DRIVE_BASE = "/content/drive/MyDrive/tower_detection"

# If running locally:
LOCAL_BASE = os.path.dirname(os.path.abspath(__file__))

# Auto-detect environment
IN_COLAB = "COLAB_GPU" in os.environ or os.path.exists("/content")

if IN_COLAB:
    BASE_DIR = Path(COLAB_DRIVE_BASE)
    DATASET_YAML = BASE_DIR / "dataset_merged" / "data.yaml"
    PROJECT_DIR = "/content/runs"  # faster on Colab local storage
else:
    BASE_DIR = Path(LOCAL_BASE)
    DATASET_YAML = BASE_DIR / "dataset_merged" / "data.yaml"
    PROJECT_DIR = str(BASE_DIR / "runs")

# Training hyperparameters
MODEL_BASE = "yolov8n.pt"       # YOLOv8-nano (fastest, good for small datasets)
EPOCHS = 100
IMGSZ = 640
BATCH_SIZE = 8                  # GTX 1650 4GB VRAM - safe batch size
PATIENCE = 20                   # Early stopping patience
LR0 = 0.01                     # Initial learning rate
LRF = 0.01                     # Final learning rate factor (lr0 * lrf at end)

# =====================================================


def print_system_info():
    print("=" * 65)
    print("  TOWER COMPONENT DETECTION - YOLO TRAINING")
    print("=" * 65)
    print(f"  Environment : {'Google Colab' if IN_COLAB else 'Local'}")
    print(f"  PyTorch     : {torch.__version__}")
    print(f"  CUDA        : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU         : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM        : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  Dataset YAML: {DATASET_YAML}")
    print(f"  Base Model  : {MODEL_BASE}")
    print(f"  Epochs      : {EPOCHS}")
    print(f"  Image Size  : {IMGSZ}")
    print(f"  Batch Size  : {BATCH_SIZE}")
    print("=" * 65)


def validate_dataset():
    """Check that dataset exists and has images."""
    if not DATASET_YAML.exists():
        print(f"\nERROR: Dataset YAML not found at {DATASET_YAML}")
        print("Make sure dataset_merged/ is in the correct location.")
        if IN_COLAB:
            print(f"Expected path: {COLAB_DRIVE_BASE}/dataset_merged/data.yaml")
            print("Upload dataset_merged/ folder to your Google Drive.")
        sys.exit(1)
    
    yaml_dir = DATASET_YAML.parent
    for split in ["train", "val"]:
        img_dir = yaml_dir / "images" / split
        if not img_dir.exists():
            print(f"ERROR: {img_dir} not found!")
            sys.exit(1)
        count = len(list(img_dir.glob("*")))
        print(f"  {split}: {count} images found")
    
    print("  Dataset validation passed!")


def train():
    """Run YOLO training."""
    device = 0 if torch.cuda.is_available() else "cpu"
    
    model = YOLO(MODEL_BASE)
    
    results = model.train(
        data=str(DATASET_YAML),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH_SIZE,
        patience=PATIENCE,
        project=PROJECT_DIR,
        name="tower_final",
        device=device,
        exist_ok=True,
        
        # Optimizer
        optimizer="AdamW",
        lr0=LR0,
        lrf=LRF,
        weight_decay=0.0005,
        warmup_epochs=5,
        warmup_momentum=0.8,
        
        # Augmentation (aggressive for small datasets)
        augment=True,
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.1,
        flipud=0.5,
        fliplr=0.5,
        degrees=20.0,
        translate=0.2,
        scale=0.5,
        shear=5.0,
        perspective=0.001,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        erasing=0.3,
        
        # Other
        plots=True,
        save=True,
        save_period=10,         # Save checkpoint every 10 epochs
        val=True,
        verbose=True,
        workers=0,
        seed=42,
    )
    
    return results


def evaluate(weights_path):
    """Evaluate the trained model and save metrics."""
    print("\n" + "=" * 65)
    print("  EVALUATING BEST MODEL ON VALIDATION SET")
    print("=" * 65)
    
    model = YOLO(weights_path)
    metrics = model.val(data=str(DATASET_YAML), split="val", plots=True)
    
    summary = {
        "mAP50": round(float(metrics.box.map50), 4),
        "mAP50_95": round(float(metrics.box.map), 4),
        "precision": round(float(metrics.box.mp), 4),
        "recall": round(float(metrics.box.mr), 4),
        "weights": str(weights_path),
        "classes": {}
    }
    
    names = model.names
    for i, name in names.items():
        if i < len(metrics.box.p):
            summary["classes"][name] = {
                "precision": round(float(metrics.box.p[i]), 4),
                "recall": round(float(metrics.box.r[i]), 4),
                "mAP50": round(float(metrics.box.ap50[i]), 4),
                "mAP50_95": round(float(metrics.box.all_ap[i].mean()), 4) if hasattr(metrics.box, 'all_ap') else None
            }
    
    print(f"\n  mAP@50      : {summary['mAP50']*100:.2f}%")
    print(f"  mAP@50-95   : {summary['mAP50_95']*100:.2f}%")
    print(f"  Precision   : {summary['precision']*100:.2f}%")
    print(f"  Recall      : {summary['recall']*100:.2f}%")
    for cls_name, cls_m in summary["classes"].items():
        print(f"  [{cls_name}] P={cls_m['precision']:.3f} R={cls_m['recall']:.3f} mAP50={cls_m['mAP50']:.3f}")
    
    metrics_path = Path(PROJECT_DIR) / "tower_final" / "evaluation_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Metrics saved to: {metrics_path}")
    
    return summary


def generate_graphs(run_dir):
    """Generate training metric plots."""
    import csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    results_csv = Path(run_dir) / "results.csv"
    if not results_csv.exists():
        print("Warning: results.csv not found, skipping graph generation")
        return
    
    # Parse CSV
    data = {}
    with open(results_csv, "r") as f:
        reader = csv.reader(f)
        header = [c.strip() for c in next(reader)]
        for col in header:
            data[col] = []
        for row in reader:
            for col, val in zip(header, row):
                try:
                    data[col].append(float(val.strip()))
                except ValueError:
                    data[col].append(0)
    
    epochs = data.get("epoch", list(range(len(next(iter(data.values()))))))
    graphs_dir = Path(run_dir) / "custom_graphs"
    graphs_dir.mkdir(exist_ok=True)
    
    # 1. Loss curves
    plt.figure(figsize=(10, 6), dpi=150)
    plt.grid(True, linestyle="--", alpha=0.6)
    for col, label, color in [
        ("train/box_loss", "Train Box Loss", "#2563eb"),
        ("val/box_loss", "Val Box Loss", "#3b82f6"),
        ("train/cls_loss", "Train Cls Loss", "#ea580c"),
        ("val/cls_loss", "Val Cls Loss", "#f97316"),
        ("train/dfl_loss", "Train DFL Loss", "#16a34a"),
    ]:
        if col in data:
            ls = "--" if "val" in col else "-"
            plt.plot(epochs, data[col], label=label, color=color, linewidth=2, linestyle=ls)
    plt.title("Training & Validation Loss Curves", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
    plt.tight_layout()
    plt.savefig(graphs_dir / "training_losses.png")
    plt.close()
    
    # 2. Metrics curves
    plt.figure(figsize=(10, 6), dpi=150)
    plt.grid(True, linestyle="--", alpha=0.6)
    for col, label, color in [
        ("metrics/precision(B)", "Precision", "#0284c7"),
        ("metrics/recall(B)", "Recall", "#10b981"),
        ("metrics/mAP50(B)", "mAP@0.50", "#7c3aed"),
        ("metrics/mAP50-95(B)", "mAP@0.50-0.95", "#f59e0b"),
    ]:
        if col in data:
            ls = "--" if "95" in col else "-"
            lw = 2.5 if "mAP50(B)" == col.split("/")[-1] else 2
            plt.plot(epochs, data[col], label=label, color=color, linewidth=lw, linestyle=ls)
    plt.title("Detection Metrics Over Training", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch"); plt.ylabel("Score"); plt.ylim(0, 1.02); plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(graphs_dir / "detection_metrics.png")
    plt.close()
    
    # 3. Overview 4-panel
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
    fig.suptitle("Tower Detection Model -- Full Training Overview", fontsize=16, fontweight="bold")
    for ax in axes.flat:
        ax.grid(True, linestyle="--", alpha=0.5)
    
    if "train/box_loss" in data:
        axes[0,0].plot(epochs, data["train/box_loss"], label="Train", color="#2563eb", lw=2)
        if "val/box_loss" in data:
            axes[0,0].plot(epochs, data["val/box_loss"], label="Val", color="#3b82f6", ls="--", lw=2)
        axes[0,0].set_title("Box Regression Loss"); axes[0,0].legend()
    
    if "train/cls_loss" in data:
        axes[0,1].plot(epochs, data["train/cls_loss"], label="Train", color="#ea580c", lw=2)
        if "val/cls_loss" in data:
            axes[0,1].plot(epochs, data["val/cls_loss"], label="Val", color="#f97316", ls="--", lw=2)
        axes[0,1].set_title("Classification Loss"); axes[0,1].legend()
    
    if "metrics/mAP50(B)" in data:
        axes[1,0].plot(epochs, data["metrics/mAP50(B)"], label="mAP50", color="#7c3aed", lw=2.5)
        if "metrics/mAP50-95(B)" in data:
            axes[1,0].plot(epochs, data["metrics/mAP50-95(B)"], label="mAP50-95", color="#f59e0b", ls="--", lw=2)
        axes[1,0].set_title("Mean Average Precision"); axes[1,0].set_ylim(0, 1.02); axes[1,0].legend()
    
    if "metrics/precision(B)" in data:
        axes[1,1].plot(epochs, data["metrics/precision(B)"], label="Precision", color="#0284c7", lw=2)
        if "metrics/recall(B)" in data:
            axes[1,1].plot(epochs, data["metrics/recall(B)"], label="Recall", color="#10b981", lw=2)
        axes[1,1].set_title("Precision & Recall"); axes[1,1].set_ylim(0, 1.02); axes[1,1].legend()
    
    plt.tight_layout()
    plt.savefig(graphs_dir / "training_overview.png")
    plt.close()
    
    print(f"\n  Custom graphs saved to: {graphs_dir}")


def main():
    print_system_info()
    validate_dataset()
    
    print("\n  Starting training...")
    results = train()
    
    # Find best weights
    best_weights = Path(PROJECT_DIR) / "tower_final" / "weights" / "best.pt"
    if not best_weights.exists():
        candidates = list(Path(PROJECT_DIR).rglob("best.pt"))
        if candidates:
            best_weights = candidates[0]
    
    print(f"\n  Best weights saved at: {best_weights}")
    
    # Evaluate
    summary = evaluate(str(best_weights))
    
    # Generate graphs
    run_dir = Path(PROJECT_DIR) / "tower_final"
    if not (run_dir / "results.csv").exists():
        candidates = list(Path(PROJECT_DIR).rglob("results.csv"))
        if candidates:
            run_dir = candidates[0].parent
    generate_graphs(str(run_dir))
    
    # Copy best weights to a convenient location
    final_weights = BASE_DIR / "best_tower_model.pt"
    import shutil
    shutil.copy2(best_weights, final_weights)
    print(f"\n  Final model copied to: {final_weights}")
    
    print("\n" + "=" * 65)
    print("  TRAINING COMPLETE!")
    print("=" * 65)
    print(f"  Best model : {final_weights}")
    print(f"  mAP@50     : {summary['mAP50']*100:.2f}%")
    print(f"  Precision  : {summary['precision']*100:.2f}%")
    print(f"  Recall     : {summary['recall']*100:.2f}%")
    print("=" * 65)


if __name__ == "__main__":
    main()
