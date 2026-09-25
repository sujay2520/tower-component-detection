"""evaluate.py — Evaluate trained model and generate evidence for presentation."""
from ultralytics import YOLO
import json
from pathlib import Path

def main():
    weights_path = "runs/tower_detect/weights/best.pt"
    if not Path(weights_path).exists():
        # Try alternative paths
        alt = list(Path("runs").rglob("best.pt"))
        if alt:
            weights_path = str(alt[0])
            print(f"Using weights: {weights_path}")
        else:
            print("ERROR: No trained weights found. Run train.py first.")
            return
    
    model = YOLO(weights_path)
    metrics = model.val(data="dataset/data.yaml", split="val")

    summary = {
        "mAP50": round(float(metrics.box.map50), 4),
        "mAP50-95": round(float(metrics.box.map), 4),
        "precision": round(float(metrics.box.mp), 4),
        "recall": round(float(metrics.box.mr), 4),
    }

    print("\n" + "="*50)
    print("       EVALUATION SUMMARY")
    print("="*50)
    for k, v in summary.items():
        print(f"  {k:12s}: {v}")
    print("="*50)
    
    # Save metrics to JSON for the dashboard
    with open("evaluation_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nMetrics saved to evaluation_metrics.json")
    print("Confusion matrix & PR curves saved under runs/detect/val/")
    print("\nScreenshot these for your presentation slides!")

if __name__ == "__main__":
    main()
