"""generate_graphs.py — Generate high-resolution training & validation metric charts without pandas."""
import os
import csv
import shutil
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

def parse_results_csv(csv_path):
    """Parse results.csv into dictionary of lists using standard library."""
    data = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = [col.strip() for col in next(reader)]
        for col in header:
            data[col] = []
        for row in reader:
            if not row:
                continue
            for col, val in zip(header, row):
                try:
                    data[col].append(float(val.strip()))
                except ValueError:
                    data[col].append(val.strip())
    return data

def generate_training_plots(run_dir="runs/tower_roboflow", output_dir="training_graphs"):
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
            print(f"Warning: results.csv not found in {run_dir}")
            return
    
    print(f"Loading training log from {csv_file}")
    data = parse_results_csv(csv_file)
    epochs = data.get('epoch', list(range(1, len(next(iter(data.values()))) + 1)))
    
    # 1. Training Losses Plot
    plt.figure(figsize=(10, 6), dpi=300)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    if 'train/box_loss' in data:
        plt.plot(epochs, data['train/box_loss'], label='Train Box Loss', color='#2563eb', linewidth=2)
    if 'val/box_loss' in data:
        plt.plot(epochs, data['val/box_loss'], label='Val Box Loss', color='#3b82f6', linestyle='--', linewidth=2)
    if 'train/cls_loss' in data:
        plt.plot(epochs, data['train/cls_loss'], label='Train Class Loss', color='#ea580c', linewidth=2)
    if 'val/cls_loss' in data:
        plt.plot(epochs, data['val/cls_loss'], label='Val Class Loss', color='#f97316', linestyle='--', linewidth=2)
    if 'train/dfl_loss' in data:
        plt.plot(epochs, data['train/dfl_loss'], label='Train DFL Loss', color='#16a34a', linewidth=1.5)
    
    plt.title('Tower Detection Model — Training & Validation Loss', fontsize=14, fontweight='bold', pad=12)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss Value', fontsize=12)
    plt.legend(frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    loss_path = out_path / "training_losses.png"
    plt.savefig(loss_path)
    plt.close()
    print(f"Saved: {loss_path}")
    
    # 2. Detection Metrics Plot (Precision, Recall, mAP50, mAP50-95)
    plt.figure(figsize=(10, 6), dpi=300)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    if 'metrics/precision(B)' in data:
        plt.plot(epochs, data['metrics/precision(B)'], label='Precision', color='#0284c7', linewidth=2)
    if 'metrics/recall(B)' in data:
        plt.plot(epochs, data['metrics/recall(B)'], label='Recall', color='#10b981', linewidth=2)
    if 'metrics/mAP50(B)' in data:
        plt.plot(epochs, data['metrics/mAP50(B)'], label='mAP @ 0.50', color='#7c3aed', linewidth=2.5)
    if 'metrics/mAP50-95(B)' in data:
        plt.plot(epochs, data['metrics/mAP50-95(B)'], label='mAP @ 0.50-0.95', color='#f59e0b', linestyle='--', linewidth=2)
    
    plt.title('Model Accuracy Progression (mAP, Precision, Recall)', fontsize=14, fontweight='bold', pad=12)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Metric Score (0.0 - 1.0)', fontsize=12)
    plt.ylim(0, 1.02)
    plt.legend(frameon=True, facecolor='white', framealpha=0.9, loc='lower right')
    plt.tight_layout()
    metrics_path = out_path / "detection_metrics.png"
    plt.savefig(metrics_path)
    plt.close()
    print(f"Saved: {metrics_path}")
    
    # 3. 4-Panel Training Overview Dashboard
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
    fig.suptitle('AI Tower Component Detection — Model Performance Overview', fontsize=16, fontweight='bold')
    
    for ax in axes.flat:
        ax.grid(True, linestyle='--', alpha=0.6)
    
    # Panel A: Box Loss
    if 'train/box_loss' in data and 'val/box_loss' in data:
        axes[0, 0].plot(epochs, data['train/box_loss'], label='Train Box Loss', color='#2563eb', lw=2)
        axes[0, 0].plot(epochs, data['val/box_loss'], label='Val Box Loss', color='#3b82f6', ls='--', lw=2)
        axes[0, 0].set_title('Bounding Box Regression Loss', fontweight='semibold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Box Loss')
        axes[0, 0].legend()
    
    # Panel B: Class Loss
    if 'train/cls_loss' in data and 'val/cls_loss' in data:
        axes[0, 1].plot(epochs, data['train/cls_loss'], label='Train Class Loss', color='#ea580c', lw=2)
        axes[0, 1].plot(epochs, data['val/cls_loss'], label='Val Class Loss', color='#f97316', ls='--', lw=2)
        axes[0, 1].set_title('Classification Cross-Entropy Loss', fontweight='semibold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Cls Loss')
        axes[0, 1].legend()
    
    # Panel C: mAP progression
    if 'metrics/mAP50(B)' in data and 'metrics/mAP50-95(B)' in data:
        axes[1, 0].plot(epochs, data['metrics/mAP50(B)'], label='mAP50', color='#7c3aed', lw=2.5)
        axes[1, 0].plot(epochs, data['metrics/mAP50-95(B)'], label='mAP50-95', color='#f59e0b', ls='--', lw=2)
        axes[1, 0].set_title('Mean Average Precision (mAP)', fontweight='semibold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('mAP')
        axes[1, 0].set_ylim(0, 1.02)
        axes[1, 0].legend()
    
    # Panel D: Precision & Recall
    if 'metrics/precision(B)' in data and 'metrics/recall(B)' in data:
        axes[1, 1].plot(epochs, data['metrics/precision(B)'], label='Precision', color='#0284c7', lw=2)
        axes[1, 1].plot(epochs, data['metrics/recall(B)'], label='Recall', color='#10b981', lw=2)
        axes[1, 1].set_title('Precision & Recall Curves', fontweight='semibold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_ylim(0, 1.02)
        axes[1, 1].legend()
    
    plt.tight_layout()
    overview_path = out_path / "training_overview.png"
    plt.savefig(overview_path)
    plt.close()
    print(f"Saved: {overview_path}")
    
    # Copy Ultralytics auto-generated plots if available
    auto_files = [
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "BoxPR_curve.png",
        "BoxF1_curve.png",
        "val_batch0_pred.jpg",
        "results.png"
    ]
    for af in auto_files:
        src = run_path / af
        if src.exists():
            shutil.copy(src, out_path / af)
            print(f"Copied {af} to {out_path}")

if __name__ == "__main__":
    generate_training_plots()
