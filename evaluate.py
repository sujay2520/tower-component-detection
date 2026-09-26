"""evaluate.py — Evaluate Roboflow-trained tower detection model and save metrics."""
import json
from pathlib import Path
from ultralytics import YOLO

def main():
    candidate_weights = [
        "runs/detect/runs/tower_roboflow/weights/best.pt",
        "runs/tower_roboflow/weights/best.pt",
        "runs/detect/runs/tower_detect/weights/best.pt"
    ]
    weights_path = None
    for p in candidate_weights:
        if Path(p).exists():
            weights_path = p
            break
            
    if not weights_path:
        alt = list(Path("runs").rglob("best.pt"))
        if alt:
            weights_path = str(alt[0])
            
    if not weights_path:
        print("ERROR: No trained weights found.")
        return

    print(f"Loading weights from: {weights_path}")
    model = YOLO(weights_path)
    metrics = model.val(data="dataset/roboflow_data.yaml", split="val")

    summary = {
        "mAP50": round(float(metrics.box.map50), 4),
        "mAP50_95": round(float(metrics.box.map), 4),
        "precision": round(float(metrics.box.mp), 4),
        "recall": round(float(metrics.box.mr), 4),
        "weights_evaluated": weights_path,
        "classes": {}
    }

    names = model.names
    for i, name in names.items():
        if i < len(metrics.box.p):
            summary["classes"][name] = {
                "precision": round(float(metrics.box.p[i]), 4),
                "recall": round(float(metrics.box.r[i]), 4),
                "mAP50": round(float(metrics.box.ap50[i]), 4),
            }

    print("\n" + "=" * 55)
    print("           EVALUATION SUMMARY")
    print("=" * 55)
    print(f"  mAP @ 0.50     : {summary['mAP50'] * 100:.2f}%")
    print(f"  mAP @ 0.50-0.95: {summary['mAP50_95'] * 100:.2f}%")
    print(f"  Precision      : {summary['precision'] * 100:.2f}%")
    print(f"  Recall         : {summary['recall'] * 100:.2f}%")
    for name, cmetrics in summary["classes"].items():
        print(f"  [{name}] P: {cmetrics['precision']:.3f}, R: {cmetrics['recall']:.3f}, mAP50: {cmetrics['mAP50']:.3f}")
    print("=" * 55)

    with open("evaluation_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nMetrics successfully saved to evaluation_metrics.json")

if __name__ == "__main__":
    main()
