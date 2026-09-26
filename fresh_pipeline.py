"""
fresh_pipeline.py — Complete CLEAN Pipeline for Tower Detection
================================================================
Builds the entire dataset from scratch using ONLY:
  1. Images from C:\\Users\\sujay\\Downloads\\sample\\sample
  2. Manual LabelImg annotations from C:\\Users\\sujay\\Downloads\\support.zip

Pipeline Steps:
  Step 1: Extract LabelImg XML annotations from support.zip
  Step 2: Match XMLs to images (only use images that have manual annotations)
  Step 3: Quality filter — reject blurred, underexposed, overexposed images
  Step 4: Convert Pascal VOC XML → YOLO txt format
  Step 5: Albumentations augmentations (rotation, brightness, contrast, exposure, angles)
  Step 6: Train/Val split (85/15)
  Step 7: Create YOLO data.yaml
  Step 8: Train YOLOv8n for 86 epochs

Classes:
  0 = supporting_tower
  1 = monopole_tower
"""

import os
import sys
import shutil
import zipfile
import random
import csv
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter

import cv2
import numpy as np

try:
    import albumentations as A
except ImportError:
    os.system("pip install albumentations")
    import albumentations as A

# ============================================================
# CONFIGURATION
# ============================================================
SAMPLE_DIR = Path(r"C:\Users\sujay\Downloads\sample\sample")
SUPPORT_ZIP = Path(r"C:\Users\sujay\Downloads\support.zip")
PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "dataset_fresh"
ANNOTATIONS_DIR = PROJECT_ROOT / "annotations_extracted"

CLASS_MAP = {"supporting_tower": 0, "monopole_tower": 1}
CLASS_NAMES = ["supporting_tower", "monopole_tower"]

# Quality thresholds
MIN_RESOLUTION = 100          # minimum dimension in pixels
BLUR_THRESHOLD = 80.0         # Laplacian variance < this = too blurry
BRIGHTNESS_MIN = 25.0         # mean pixel < this = underexposed
BRIGHTNESS_MAX = 240.0        # mean pixel > this = overexposed

# Augmentation
NUM_AUG_COPIES = 3            # augmented copies per original image

# Training
EPOCHS = 86
BATCH_SIZE = 8
VAL_SPLIT = 0.15

random.seed(42)
np.random.seed(42)


# ============================================================
# STEP 1: Extract annotations from support.zip
# ============================================================
def step1_extract_annotations():
    print("\n" + "=" * 60)
    print("  STEP 1: Extracting LabelImg annotations from support.zip")
    print("=" * 60)

    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(str(SUPPORT_ZIP), 'r') as zf:
        xml_count = 0
        for entry in zf.namelist():
            if entry.endswith('.xml'):
                # Extract to flat directory
                fname = os.path.basename(entry)
                target = ANNOTATIONS_DIR / fname
                with zf.open(entry) as src, open(target, 'wb') as dst:
                    dst.write(src.read())
                xml_count += 1
        # Also extract classes.txt if present
        for entry in zf.namelist():
            if entry.endswith('classes.txt'):
                fname = os.path.basename(entry)
                with zf.open(entry) as src, open(ANNOTATIONS_DIR / fname, 'wb') as dst:
                    dst.write(src.read())

    print(f"  Extracted {xml_count} XML annotation files")
    return xml_count


# ============================================================
# STEP 2: Match annotations to images
# ============================================================
def step2_match_annotations():
    print("\n" + "=" * 60)
    print("  STEP 2: Matching annotations to images")
    print("=" * 60)

    # Build image lookup (case-insensitive stem → path)
    image_lookup = {}
    for ext in ["*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.png", "*.PNG", "*.bmp", "*.webp"]:
        for img_path in SAMPLE_DIR.glob(ext):
            stem = img_path.stem.lower()
            image_lookup[stem] = img_path

    print(f"  Found {len(image_lookup)} images in sample directory")

    # Match XMLs to images
    matched = []
    unmatched_xmls = []
    class_counts = Counter()

    for xml_path in sorted(ANNOTATIONS_DIR.glob("*.xml")):
        stem = xml_path.stem.lower()
        if stem in image_lookup:
            # Parse XML to get class
            tree = ET.parse(str(xml_path))
            root = tree.getroot()
            objects = root.findall("object")
            classes_in_file = [obj.find("name").text for obj in objects if obj.find("name") is not None]

            # Only keep tower classes
            tower_classes = [c for c in classes_in_file if c in CLASS_MAP]
            if tower_classes:
                matched.append({
                    "image_path": image_lookup[stem],
                    "xml_path": xml_path,
                    "classes": tower_classes
                })
                for c in tower_classes:
                    class_counts[c] += 1
            else:
                unmatched_xmls.append(xml_path.name)
        else:
            unmatched_xmls.append(xml_path.name)

    print(f"  Matched: {len(matched)} annotated images")
    print(f"  Unmatched XMLs: {len(unmatched_xmls)}")
    for cls, count in class_counts.items():
        print(f"    {cls}: {count} annotations")

    return matched


# ============================================================
# STEP 3: Quality filter
# ============================================================
def step3_quality_filter(matched):
    print("\n" + "=" * 60)
    print("  STEP 3: Quality filtering (blur, exposure)")
    print("=" * 60)

    passed = []
    rejected = {"blur": 0, "underexposed": 0, "overexposed": 0, "low_res": 0, "unreadable": 0}

    for item in matched:
        img = cv2.imread(str(item["image_path"]))
        if img is None:
            rejected["unreadable"] += 1
            print(f"    [REJECT] Cannot read: {item['image_path'].name}")
            continue

        h, w = img.shape[:2]

        # Resolution check
        if min(h, w) < MIN_RESOLUTION:
            rejected["low_res"] += 1
            print(f"    [REJECT] Low resolution ({w}x{h}): {item['image_path'].name}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Blur check (Laplacian variance)
        blur_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if blur_var < BLUR_THRESHOLD:
            rejected["blur"] += 1
            print(f"    [REJECT] Blurry (variance={blur_var:.1f}): {item['image_path'].name}")
            continue

        # Exposure check
        brightness = float(np.mean(gray))
        if brightness < BRIGHTNESS_MIN:
            rejected["underexposed"] += 1
            print(f"    [REJECT] Underexposed (brightness={brightness:.1f}): {item['image_path'].name}")
            continue
        if brightness > BRIGHTNESS_MAX:
            rejected["overexposed"] += 1
            print(f"    [REJECT] Overexposed (brightness={brightness:.1f}): {item['image_path'].name}")
            continue

        passed.append(item)

    print(f"\n  Passed quality check: {len(passed)}/{len(matched)}")
    for reason, count in rejected.items():
        if count > 0:
            print(f"    Rejected ({reason}): {count}")

    return passed


# ============================================================
# STEP 4: Convert Pascal VOC XML → YOLO txt format
# ============================================================
def parse_voc_xml(xml_path, img_w, img_h):
    """Parse Pascal VOC XML and return list of YOLO-format annotations."""
    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    yolo_lines = []
    bboxes = []
    class_ids = []

    for obj in root.findall("object"):
        name = obj.find("name").text
        if name not in CLASS_MAP:
            continue

        class_id = CLASS_MAP[name]
        bndbox = obj.find("bndbox")
        xmin = max(0, int(float(bndbox.find("xmin").text)))
        ymin = max(0, int(float(bndbox.find("ymin").text)))
        xmax = min(img_w, int(float(bndbox.find("xmax").text)))
        ymax = min(img_h, int(float(bndbox.find("ymax").text)))

        # YOLO format: class x_center y_center width height (normalized 0-1)
        x_center = ((xmin + xmax) / 2.0) / img_w
        y_center = ((ymin + ymax) / 2.0) / img_h
        bbox_w = (xmax - xmin) / img_w
        bbox_h = (ymax - ymin) / img_h

        # Validate
        if bbox_w <= 0 or bbox_h <= 0:
            continue
        if x_center < 0 or x_center > 1 or y_center < 0 or y_center > 1:
            continue

        yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_w:.6f} {bbox_h:.6f}")
        bboxes.append([xmin, ymin, xmax, ymax])
        class_ids.append(class_id)

    return yolo_lines, bboxes, class_ids


# ============================================================
# STEP 5: Albumentations augmentations
# ============================================================
def get_augmentation_pipeline():
    """Create Albumentations augmentation pipeline with diverse transforms."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=20, border_mode=cv2.BORDER_REFLECT_101, p=0.6),
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.7),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=25, val_shift_limit=25, p=0.5),
        A.OneOf([
            A.MotionBlur(blur_limit=5, p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
        ], p=0.25),
        A.OneOf([
            A.RandomGamma(gamma_limit=(70, 130), p=1.0),
            A.CLAHE(clip_limit=3.0, p=1.0),
        ], p=0.3),
        A.RandomScale(scale_limit=0.15, p=0.3),
        A.PadIfNeeded(min_height=640, min_width=640, border_mode=cv2.BORDER_REFLECT_101, p=0.0),
    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['class_ids'],
        min_visibility=0.3
    ))


def augment_image(image, bboxes, class_ids, transform, max_dim=1280):
    """Apply augmentation and return transformed image + bboxes."""
    h, w = image.shape[:2]

    # Resize large images to prevent memory issues
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        # Scale bboxes
        bboxes = [[int(b[0]*scale), int(b[1]*scale), int(b[2]*scale), int(b[3]*scale)] for b in bboxes]
        h, w = new_h, new_w

    # Clamp bboxes to image bounds
    clamped_bboxes = []
    clamped_ids = []
    for bbox, cid in zip(bboxes, class_ids):
        x1 = max(0, min(bbox[0], w - 1))
        y1 = max(0, min(bbox[1], h - 1))
        x2 = max(x1 + 1, min(bbox[2], w))
        y2 = max(y1 + 1, min(bbox[3], h))
        if x2 - x1 > 5 and y2 - y1 > 5:
            clamped_bboxes.append([x1, y1, x2, y2])
            clamped_ids.append(cid)

    if not clamped_bboxes:
        return None, None, None

    try:
        transformed = transform(image=image, bboxes=clamped_bboxes, class_ids=clamped_ids)
        return transformed['image'], transformed['bboxes'], transformed['class_ids']
    except Exception:
        return None, None, None


# ============================================================
# STEP 6-7: Build YOLO dataset with train/val split
# ============================================================
def step4to7_build_dataset(passed_items):
    print("\n" + "=" * 60)
    print("  STEPS 4-7: Building YOLO dataset with augmentations")
    print("=" * 60)

    # Create dataset directories
    for split in ["train", "val"]:
        (DATASET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Stratified train/val split — ensure both classes appear in val
    supporting_items = [it for it in passed_items if "supporting_tower" in it["classes"]]
    monopole_items = [it for it in passed_items if "monopole_tower" in it["classes"] and it not in supporting_items]
    random.shuffle(supporting_items)
    random.shuffle(monopole_items)

    # Take 15% from each class for val (minimum 2 per class)
    s_val_n = max(2, int(len(supporting_items) * VAL_SPLIT))
    m_val_n = max(2, int(len(monopole_items) * VAL_SPLIT))

    val_items = supporting_items[:s_val_n] + monopole_items[:m_val_n]
    train_items = supporting_items[s_val_n:] + monopole_items[m_val_n:]
    random.shuffle(val_items)
    random.shuffle(train_items)

    print(f"  Train: {len(train_items)} images | Val: {len(val_items)} images")

    transform = get_augmentation_pipeline()
    stats = {"train_orig": 0, "train_aug": 0, "val_orig": 0, "class_train": Counter(), "class_val": Counter()}

    # Process validation set (NO augmentation)
    print("\n  Processing validation set (no augmentation)...")
    for item in val_items:
        img = cv2.imread(str(item["image_path"]))
        if img is None:
            continue
        orig_h, orig_w = img.shape[:2]

        # Parse XML using original image dimensions (XML bboxes are in original pixel coords)
        yolo_lines, bboxes, class_ids = parse_voc_xml(item["xml_path"], orig_w, orig_h)
        if not yolo_lines:
            continue

        # Resize for saving
        save_img = img.copy()
        if max(orig_h, orig_w) > 1280:
            scale = 1280 / max(orig_h, orig_w)
            save_img = cv2.resize(save_img, (int(orig_w * scale), int(orig_h * scale)), interpolation=cv2.INTER_AREA)

        stem = item["image_path"].stem
        img_out = DATASET_DIR / "images" / "val" / f"{stem}.jpg"
        lbl_out = DATASET_DIR / "labels" / "val" / f"{stem}.txt"

        cv2.imwrite(str(img_out), save_img)
        # YOLO format is normalized 0-1, so labels are resolution-independent
        with open(lbl_out, "w") as f:
            f.write("\n".join(yolo_lines))

        stats["val_orig"] += 1
        for cid in class_ids:
            stats["class_val"][CLASS_NAMES[cid]] += 1

    # Process training set (originals + augmentations)
    print("  Processing training set (originals + augmentations)...")
    for idx, item in enumerate(train_items):
        img = cv2.imread(str(item["image_path"]))
        if img is None:
            continue
        orig_h, orig_w = img.shape[:2]

        yolo_lines, bboxes, class_ids = parse_voc_xml(item["xml_path"], orig_w, orig_h)
        if not yolo_lines:
            continue

        stem = item["image_path"].stem

        # Save original (resized)
        save_img = img.copy()
        sh, sw = save_img.shape[:2]
        if max(sh, sw) > 1280:
            scale = 1280 / max(sh, sw)
            save_img = cv2.resize(save_img, (int(sw * scale), int(sh * scale)), interpolation=cv2.INTER_AREA)

        img_out = DATASET_DIR / "images" / "train" / f"{stem}.jpg"
        lbl_out = DATASET_DIR / "labels" / "train" / f"{stem}.txt"
        cv2.imwrite(str(img_out), save_img)
        with open(lbl_out, "w") as f:
            f.write("\n".join(yolo_lines))
        stats["train_orig"] += 1
        for cid in class_ids:
            stats["class_train"][CLASS_NAMES[cid]] += 1

        # Generate augmented copies
        for aug_idx in range(NUM_AUG_COPIES):
            aug_img, aug_bboxes, aug_ids = augment_image(img, bboxes, class_ids, transform)
            if aug_img is None or not aug_bboxes:
                continue

            ah, aw = aug_img.shape[:2]
            aug_yolo = []
            for bbox, cid in zip(aug_bboxes, aug_ids):
                x1, y1, x2, y2 = bbox
                xc = ((x1 + x2) / 2.0) / aw
                yc = ((y1 + y2) / 2.0) / ah
                bw = (x2 - x1) / aw
                bh = (y2 - y1) / ah
                if 0 <= xc <= 1 and 0 <= yc <= 1 and bw > 0 and bh > 0:
                    aug_yolo.append(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

            if not aug_yolo:
                continue

            aug_stem = f"{stem}_aug{aug_idx}"
            aug_img_out = DATASET_DIR / "images" / "train" / f"{aug_stem}.jpg"
            aug_lbl_out = DATASET_DIR / "labels" / "train" / f"{aug_stem}.txt"
            cv2.imwrite(str(aug_img_out), aug_img)
            with open(aug_lbl_out, "w") as f:
                f.write("\n".join(aug_yolo))
            stats["train_aug"] += 1

        if (idx + 1) % 20 == 0:
            print(f"    Processed {idx + 1}/{len(train_items)} training images...")

    # Create data.yaml
    data_yaml = DATASET_DIR / "data.yaml"
    with open(data_yaml, "w") as f:
        f.write(f"path: {DATASET_DIR}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write(f"\nnc: {len(CLASS_NAMES)}\n")
        f.write(f"names: {CLASS_NAMES}\n")

    total_train = stats["train_orig"] + stats["train_aug"]
    total_val = stats["val_orig"]

    print(f"\n  Dataset created at: {DATASET_DIR}")
    print(f"  Train: {stats['train_orig']} originals + {stats['train_aug']} augmented = {total_train} total")
    print(f"  Val: {total_val}")
    print(f"  Train class distribution: {dict(stats['class_train'])}")
    print(f"  Val class distribution: {dict(stats['class_val'])}")

    return total_train, total_val


# ============================================================
# STEP 8: Train YOLOv8
# ============================================================
def step8_train():
    print("\n" + "=" * 60)
    print("  STEP 8: Training YOLOv8n for 86 epochs")
    print("=" * 60)

    import torch
    from ultralytics import YOLO

    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    data_yaml = str(DATASET_DIR / "data.yaml")
    device = 0 if torch.cuda.is_available() else "cpu"

    model = YOLO("yolov8n.pt")

    results = model.train(
        data=data_yaml,
        epochs=EPOCHS,
        imgsz=640,
        batch=BATCH_SIZE,
        patience=0,
        project=str(PROJECT_ROOT / "runs"),
        name="tower_fresh",
        device=device,
        exist_ok=True,
        agnostic_nms=True,
        iou=0.3,

        # Optimizer
        optimizer="AdamW",
        lr0=0.01,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=4,

        # Built-in YOLO augmentations (on top of our Albumentations)
        augment=True,
        mosaic=0.8,
        mixup=0.10,
        flipud=0.3,
        fliplr=0.5,

        plots=True,
        save=True,
        save_period=10,
        val=True,
        workers=0,
        seed=42
    )

    # Find best weights
    best_pt = PROJECT_ROOT / "runs" / "tower_fresh" / "weights" / "best.pt"
    if not best_pt.exists():
        candidates = list((PROJECT_ROOT / "runs").rglob("best.pt"))
        if candidates:
            best_pt = candidates[0]

    print(f"\n  Best weights: {best_pt}")

    # Evaluate
    eval_model = YOLO(str(best_pt))
    metrics = eval_model.val(data=data_yaml, split="val", plots=True)

    summary = {
        "dataset": "Fresh LabelImg Dataset (quality filtered + augmented)",
        "model": "YOLOv8n",
        "epochs": EPOCHS,
        "mAP50": round(float(metrics.box.map50), 4),
        "mAP50_95": round(float(metrics.box.map), 4),
        "precision": round(float(metrics.box.mp), 4),
        "recall": round(float(metrics.box.mr), 4),
        "weights": str(best_pt),
        "classes": {}
    }

    for i, name in eval_model.names.items():
        if i < len(metrics.box.p):
            summary["classes"][name] = {
                "precision": round(float(metrics.box.p[i]), 4),
                "recall": round(float(metrics.box.r[i]), 4),
                "mAP50": round(float(metrics.box.ap50[i]), 4),
            }

    with open(PROJECT_ROOT / "evaluation_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Final mAP@50   : {summary['mAP50']*100:.2f}%")
    print(f"  Final mAP50-95 : {summary['mAP50_95']*100:.2f}%")
    print(f"  Precision      : {summary['precision']*100:.2f}%")
    print(f"  Recall         : {summary['recall']*100:.2f}%")
    for name, info in summary["classes"].items():
        print(f"  [{name}] P={info['precision']:.3f}, R={info['recall']:.3f}, mAP50={info['mAP50']:.3f}")

    # Copy best weights to standard locations
    shutil.copy2(str(best_pt), str(PROJECT_ROOT / "best_tower_model.pt"))
    (PROJECT_ROOT / "runs" / "tower_roboflow" / "weights").mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(best_pt), str(PROJECT_ROOT / "runs" / "tower_roboflow" / "weights" / "best.pt"))

    # Generate graphs
    generate_graphs(PROJECT_ROOT / "runs" / "tower_fresh", PROJECT_ROOT / "training_graphs")

    print("\n" + "=" * 60)
    print("  FRESH PIPELINE COMPLETE!")
    print("=" * 60)


# ============================================================
# Graph generation
# ============================================================
def generate_graphs(run_dir, output_dir):
    """Generate training metric graphs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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
            print("  results.csv not found, skipping graphs")
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

    # 4-Panel Overview
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=200)
    fig.suptitle("Tower Detection — Fresh Training Performance", fontsize=16, fontweight="bold")
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

    # Copy YOLO-generated plots
    for f in ["confusion_matrix.png", "confusion_matrix_normalized.png",
              "BoxPR_curve.png", "BoxF1_curve.png", "results.png", "val_batch0_pred.jpg"]:
        src = run_path / f
        if src.exists():
            shutil.copy2(src, out_path / f)

    print("  Training graphs generated!")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  FRESH TOWER DETECTION PIPELINE")
    print("  Using ONLY: sample images + support.zip annotations")
    print("  Model: YOLOv8n | Quality Filter: ON")
    print("=" * 60)

    step1_extract_annotations()
    matched = step2_match_annotations()
    passed = step3_quality_filter(matched)

    if len(passed) < 5:
        print(f"\nERROR: Only {len(passed)} images passed quality filter. Need at least 5.")
        sys.exit(1)

    train_count, val_count = step4to7_build_dataset(passed)

    if train_count < 3 or val_count < 1:
        print(f"\nERROR: Not enough images for training ({train_count} train, {val_count} val)")
        sys.exit(1)

    step8_train()
