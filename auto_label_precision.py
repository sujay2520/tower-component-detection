"""
auto_label_precision.py — High-precision tight bounding box auto-labeler.
Replaces flawed full-image boxes with tight, accurate bounding boxes using
the Roboflow-trained detector + contour saliency refinement.
Fixes the root cause of low detection confidence.
"""

import os
import shutil
import random
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

ROBOFLOW_DIR = Path(r"C:\Users\sujay\Downloads\annotaate.v1i.yolov11")
SUPPORTING_DIR = Path(r"C:\Users\sujay\Downloads\supporting (2)\supporting\supporting")
MONO_DIR = Path(r"C:\Users\sujay\Downloads\Mono")
SAMPLE_DIR = Path(r"C:\Users\sujay\Downloads\sample\sample")

OUTPUT_DIR = Path("dataset_merged")
CLASSES = ["monopole_tower", "supporting_tower"]

MAX_DIM = 1280
MIN_DIM = 64
SEED = 42
random.seed(SEED)


def get_tight_box_fallback(image, is_monopole=False):
    """
    Saliency-based vertical column localization fallback when detector has low confidence.
    Towers are vertical structures in the central horizontal region.
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Edge detection
    edges = cv2.Canny(blurred, 30, 120)

    # Vertical projection profile (sum along columns)
    col_density = np.sum(edges, axis=0)
    window = int(w * 0.15)
    smooth_density = np.convolve(col_density, np.ones(window)/window, mode='same')

    # Find the peak center column
    center_col = np.argmax(smooth_density)
    # Ensure center column is reasonably within frame
    if center_col < w * 0.1 or center_col > w * 0.9:
        center_col = int(w * 0.5)

    if is_monopole:
        box_w = min(max(int(w * 0.18), 30), int(w * 0.40))
        box_h = int(h * 0.88)
    else:
        box_w = min(max(int(w * 0.38), 60), int(w * 0.70))
        box_h = int(h * 0.92)

    x1 = max(0, center_col - box_w // 2)
    x2 = min(w, center_col + box_w // 2)
    y1 = max(0, int(h * 0.04))
    y2 = min(h, y1 + box_h)

    # Convert to normalized YOLO format (cx, cy, bw, bh)
    cx = ((x1 + x2) / 2.0) / w
    cy = ((y1 + y2) / 2.0) / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return f"{cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def main():
    print("=" * 60)
    print("  HIGH-PRECISION TIGHT BOUNDING BOX AUTO-LABELER")
    print("=" * 60)

    # Load Roboflow model for inference-based pseudo-labeling
    model_path = Path("runs/detect/runs/tower_roboflow/weights/best.pt")
    if not model_path.exists():
        model_path = Path("runs/tower_roboflow/weights/best.pt")
    
    print(f"Loading pseudo-labeling model from: {model_path}")
    model = YOLO(str(model_path))

    # Clean and re-create destination directories
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    
    for split in ["train", "val", "test"]:
        (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    # 1. Copy original Roboflow data untouched (already has human-drawn tight boxes)
    print("\n[1/4] Copying Roboflow annotated images and labels...")
    for split_src, split_dst in [("train", "train"), ("valid", "val"), ("test", "test")]:
        img_src = ROBOFLOW_DIR / split_src / "images"
        lbl_src = ROBOFLOW_DIR / split_src / "labels"
        if not img_src.exists():
            continue
        
        img_dst = OUTPUT_DIR / "images" / split_dst
        lbl_dst = OUTPUT_DIR / "labels" / split_dst
        
        copied = 0
        for f in img_src.iterdir():
            if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
                lbl = lbl_src / f"{f.stem}.txt"
                if lbl.exists():
                    # Resize image if needed to prevent memory issues
                    with Image.open(f) as im:
                        if im.mode in ("RGBA", "P"):
                            im = im.convert("RGB")
                        w, h = im.size
                        if max(w, h) > MAX_DIM:
                            ratio = MAX_DIM / max(w, h)
                            im = im.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
                        im.save(img_dst / f"{f.stem}.jpg", "JPEG", quality=95)
                    shutil.copy2(lbl, lbl_dst / f"{f.stem}.txt")
                    copied += 1
        print(f"  Roboflow {split_src} -> {split_dst}: {copied} images")

    # Helper function to generate tight boxes on a folder of known class
    def process_folder(folder, known_class_id, class_name):
        items = []
        if not folder.exists():
            print(f"  [SKIP] Folder not found: {folder}")
            return items

        img_files = [f for f in folder.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".JPG", ".JPEG"]]
        print(f"Processing {len(img_files)} images from {folder.name} for class {class_name}...")

        for f in img_files:
            image = cv2.imread(str(f))
            if image is None:
                continue
            h, w = image.shape[:2]
            if min(h, w) < MIN_DIM:
                continue

            # Resize if oversized
            if max(h, w) > MAX_DIM:
                ratio = MAX_DIM / max(h, w)
                image = cv2.resize(image, (int(w * ratio), int(h * ratio)), interpolation=cv2.INTER_AREA)
                h, w = image.shape[:2]

            # Try YOLO detection at low threshold to find the tower boundaries
            results = model(image, conf=0.08, verbose=False)[0]
            boxes = results.boxes

            yolo_labels = []
            if len(boxes) > 0:
                for b in boxes:
                    xyxy = b.xyxy[0].tolist()
                    x1, y1, x2, y2 = xyxy
                    # Normalize
                    cx = ((x1 + x2) / 2.0) / w
                    cy = ((y1 + y2) / 2.0) / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h
                    # Enforce the known true class!
                    yolo_labels.append(f"{known_class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            else:
                # Use vertical saliency fallback
                tight_box = get_tight_box_fallback(image, is_monopole=(known_class_id == 0))
                yolo_labels.append(f"{known_class_id} {tight_box}")

            safe_stem = f"{class_name}_{f.stem.replace(' ', '_').replace('(', '').replace(')', '')}"
            items.append({
                "image": image,
                "filename": f"{safe_stem}.jpg",
                "labels": yolo_labels
            })
        return items

    # 2. Process Supporting (2) images
    print("\n[2/4] Generating tight annotations for Supporting images...")
    supporting_items = process_folder(SUPPORTING_DIR, known_class_id=1, class_name="supporting_tower")

    # 3. Process Mono images
    print("\n[3/4] Generating tight annotations for Mono images...")
    mono_items = process_folder(MONO_DIR, known_class_id=0, class_name="monopole_tower")

    # 4. Process Sample images (mixed)
    print("\n[4/4] Generating tight annotations for Sample images...")
    sample_items = []
    if SAMPLE_DIR.exists():
        sample_files = [f for f in SAMPLE_DIR.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".JPG", ".JPEG"]]
        for f in sample_files:
            image = cv2.imread(str(f))
            if image is None:
                continue
            h, w = image.shape[:2]
            if min(h, w) < MIN_DIM:
                continue

            if max(h, w) > MAX_DIM:
                ratio = MAX_DIM / max(h, w)
                image = cv2.resize(image, (int(w * ratio), int(h * ratio)), interpolation=cv2.INTER_AREA)
                h, w = image.shape[:2]

            results = model(image, conf=0.08, verbose=False)[0]
            boxes = results.boxes

            yolo_labels = []
            if len(boxes) > 0:
                for b in boxes:
                    cls_id = int(b.cls[0])
                    # Standardize class
                    cls_raw = model.names.get(cls_id, "")
                    cid = 1 if "support" in cls_raw.lower() else 0
                    x1, y1, x2, y2 = b.xyxy[0].tolist()
                    cx = ((x1 + x2) / 2.0) / w
                    cy = ((y1 + y2) / 2.0) / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h
                    yolo_labels.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            else:
                # Default to supporting tower with vertical saliency box
                tight_box = get_tight_box_fallback(image, is_monopole=False)
                yolo_labels.append(f"1 {tight_box}")

            safe_stem = f"sample_{f.stem.replace(' ', '_').replace('(', '').replace(')', '')}"
            sample_items.append({
                "image": image,
                "filename": f"{safe_stem}.jpg",
                "labels": yolo_labels
            })
        print(f"  Processed {len(sample_items)} sample images")

    # Distribute new items into train/val/test
    all_new = supporting_items + mono_items + sample_items
    random.shuffle(all_new)
    n = len(all_new)
    n_test = max(int(n * 0.10), 1)
    n_val = max(int(n * 0.15), 1)
    n_train = n - n_val - n_test

    splits = {
        "train": all_new[:n_train],
        "val": all_new[n_train:n_train + n_val],
        "test": all_new[n_train + n_val:]
    }

    print("\nWriting new annotated images to dataset splits...")
    for split_name, split_list in splits.items():
        img_out = OUTPUT_DIR / "images" / split_name
        lbl_out = OUTPUT_DIR / "labels" / split_name
        for item in split_list:
            cv2.imwrite(str(img_out / item["filename"]), item["image"])
            with open(lbl_out / f"{Path(item['filename']).stem}.txt", "w") as lf:
                lf.write("\n".join(item["labels"]) + "\n")
        print(f"  Added {len(split_list)} images to {split_name}")

    # Write data.yaml
    yaml_content = f"""path: {OUTPUT_DIR.resolve()}
train: images/train
val: images/val
test: images/test

nc: 2
names: ['monopole_tower', 'supporting_tower']
"""
    with open(OUTPUT_DIR / "data.yaml", "w") as yf:
        yf.write(yaml_content)
    print(f"\nWritten: {OUTPUT_DIR / 'data.yaml'}")

    # Summary
    print("\n" + "=" * 60)
    print("    UPDATED MERGED DATASET (TIGHT ANNOTATIONS)")
    print("=" * 60)
    for s in ["train", "val", "test"]:
        imgs = len(list((OUTPUT_DIR / "images" / s).glob("*.jpg")))
        lbls = len(list((OUTPUT_DIR / "labels" / s).glob("*.txt")))
        print(f"  {s:6s}: {imgs} images, {lbls} label files")
    print("=" * 60)


if __name__ == "__main__":
    main()
