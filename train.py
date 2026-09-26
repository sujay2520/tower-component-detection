"""train.py — Train YOLOv8 on Roboflow annotated tower dataset with GPU acceleration."""
import os
import json
import torch
from pathlib import Path
from ultralytics import YOLO
from generate_graphs import generate_training_plots

def main():
    device = 0 if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print("=" * 60)
    print(f"Starting Training on Device: {device} ({device_name})")
    print("=" * 60)

    dataset_yaml = "dataset/roboflow_data.yaml"
    if not os.path.exists(dataset_yaml):
        raise FileNotFoundError(f"Dataset config not found at: {dataset_yaml}")

    # Load base pretrained model
    model = YOLO("yolov8n.pt")

    # Train model
    results = model.train(
        data=dataset_yaml,
        epochs=60,
        imgsz=640,
        batch=8,            # Optimal for 4GB VRAM GTX 1650
        patience=15,        # Early stopping patience
        project="runs",
        name="tower_roboflow",
        device=device,
        exist_ok=True,
        augment=True,
        lr0=0.01,
        lrf=0.001,
        mosaic=1.0,
        flipud=0.5,
        fliplr=0.5,
        degrees=15.0,
        scale=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
    )

    print("\n" + "=" * 60)
    print("Training Complete! Evaluating on Validation Set...")
    print("=" * 60)

    best_weights = "runs/tower_roboflow/weights/best.pt"
    eval_model = YOLO(best_weights)
    metrics = eval_model.val(data=dataset_yaml, split="val")

    # Extract detailed evaluation metrics
    summary = {
        "mAP50": round(float(metrics.box.map50), 4),
        "mAP50_95": round(float(metrics.box.map), 4),
        "precision": round(float(metrics.box.mp), 4),
        "recall": round(float(metrics.box.mr), 4),
        "classes": {}
    }

    names = eval_model.names
    for i, name in names.items():
        if i < len(metrics.box.p):
            summary["classes"][name] = {
                "precision": round(float(metrics.box.p[i]), 4),
                "recall": round(float(metrics.box.r[i]), 4),
                "mAP50": round(float(metrics.box.ap50[i]), 4),
            }

    print("\n" + "=" * 60)
    print("           FINAL EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  mAP@50     : {summary['mAP50'] * 100:.2f}%")
    print(f"  mAP@50-95  : {summary['mAP50_95'] * 100:.2f}%")
    print(f"  Precision  : {summary['precision'] * 100:.2f}%")
    print(f"  Recall     : {summary['recall'] * 100:.2f}%")
    for name, cmetrics in summary["classes"].items():
        print(f"  [{name}] P: {cmetrics['precision']:.3f}, R: {cmetrics['recall']:.3f}, mAP50: {cmetrics['mAP50']:.3f}")
    print("=" * 60)

    # Save metrics JSON
    with open("evaluation_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Metrics written to evaluation_metrics.json")

    # Generate graph suite
    print("Generating comprehensive training graphs...")
    generate_training_plots("runs/tower_roboflow", "training_graphs")
    print("All training metric graphs saved to training_graphs/")

if __name__ == "__main__":
    main()
