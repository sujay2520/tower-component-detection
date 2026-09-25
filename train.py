"""train.py — Train YOLOv8 model for tower detection."""
from ultralytics import YOLO

def main():
    # Start from pretrained COCO checkpoint for fast convergence
    model = YOLO("yolov8n.pt")  # nano = fastest; use yolov8s.pt if GPU is strong

    results = model.train(
        data="dataset/data.yaml",
        epochs=50,
        imgsz=640,
        batch=8,            # smaller batch for CPU
        patience=15,        # early stopping
        project="runs",
        name="tower_detect",
        device="cpu",       # CPU training (no GPU available)
        augment=True,
        lr0=0.01,
        lrf=0.001,
        mosaic=1.0,
        flipud=0.5,         # vertical flip (tower angles vary)
        fliplr=0.5,         # horizontal flip
        degrees=15.0,       # rotation augmentation
        scale=0.5,          # scale augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
    )

    print("\nTraining complete!")
    print("Best weights: runs/tower_detect/weights/best.pt")
    print("Results plots: runs/tower_detect/")

if __name__ == "__main__":
    main()
