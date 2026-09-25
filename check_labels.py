"""check_labels.py — Draws bounding boxes on sample images for visual verification."""
import cv2
import random
from pathlib import Path

CLASSES = ["supporting_tower", "monopole_tower"]
COLORS = [(0, 255, 0), (255, 0, 0)]  # green for supporting, blue for monopole
img_dir = Path("dataset/images/train")
lbl_dir = Path("dataset/labels/train")
out_dir = Path("check_output")
out_dir.mkdir(exist_ok=True)

sample = random.sample(list(img_dir.iterdir()), min(8, len(list(img_dir.iterdir()))))

for img_path in sample:
    img = cv2.imread(str(img_path))
    if img is None:
        continue
    h, w = img.shape[:2]
    lbl_path = lbl_dir / f"{img_path.stem}.txt"
    if not lbl_path.exists():
        continue
    for line in open(lbl_path):
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls_id, xc, yc, bw, bh = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)
        color = COLORS[cls_id] if cls_id < len(COLORS) else (0, 255, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        label = f"{CLASSES[cls_id]}"
        cv2.putText(img, label, (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    out_path = out_dir / f"check_{img_path.name}"
    cv2.imwrite(str(out_path), img)
    print(f"Saved {out_path}")

print(f"\nCheck images saved to {out_dir}/")
