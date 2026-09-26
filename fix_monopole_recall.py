"""
fix_monopole_recall.py — Fix the 0.33 monopole→background leak
================================================================
Uses ONLY the 124 provided images (no outside data).

The confusion matrix shows:
  - supporting_tower: 1.00 recall (perfect)
  - monopole_tower: 0.67 recall, 0.33 leaks to BACKGROUND (model can't see the pole)

Root cause: monopole is a thin vertical line against sky — few positive pixels,
and class imbalance (83 supporting vs 28 monopole annotations).

Fixes applied:
  1. COPY-PASTE AUGMENTATION — crop monopole instances from their bbox, paste onto
     different backgrounds from other images. This teaches the model that monopoles
     appear against diverse backgrounds, fixing the background-leak problem.
  2. TARGETED MONOPOLE OVERSAMPLING — 8x augmented copies of every monopole image
     (rotation, brightness, contrast, exposure, gamma, blur). Balances class count.
  3. STANDARD AUGMENTATION — 3x copies of supporting tower images.
  4. TRAIN AT imgsz=960 — monopole poles are thin, need more pixels to be visible.
  5. INFERENCE conf=0.371 — optimal F1 point from BoxF1 curve.
  6. TRAIN ALL 86 EPOCHS — no early stopping.

Classes: 0=supporting_tower, 1=monopole_tower
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
DATASET_DIR = PROJECT_ROOT / "dataset_final"
ANNOTATIONS_DIR = PROJECT_ROOT / "annotations_extracted"

CLASS_MAP = {"supporting_tower": 0, "monopole_tower": 1}
CLASS_NAMES = ["supporting_tower", "monopole_tower"]

# Quality thresholds
BLUR_THRESHOLD = 60.0
BRIGHTNESS_MIN = 20.0
BRIGHTNESS_MAX = 245.0

# Augmentation counts
MONOPOLE_AUG_COPIES = 8    # oversample monopole heavily (8x)
SUPPORTING_AUG_COPIES = 2  # light augmentation for supporting
COPY_PASTE_PER_MONOPOLE = 5  # paste each monopole crop onto 5 backgrounds

# Training
EPOCHS = 86
BATCH_SIZE = 8
IMGSZ = 960   # bigger = better for thin monopole poles
CONF_THRESHOLD = 0.371  # optimal F1 from BoxF1 curve
VAL_SPLIT = 0.15

random.seed(42)
np.random.seed(42)


# ============================================================
# STEP 1: Extract annotations
# ============================================================
def step1_extract():
    print("\n" + "=" * 60)
    print("  STEP 1: Extracting LabelImg annotations")
    print("=" * 60)
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(str(SUPPORT_ZIP), 'r') as zf:
        count = 0
        for entry in zf.namelist():
            if entry.endswith('.xml'):
                fname = os.path.basename(entry)
                with zf.open(entry) as src, open(ANNOTATIONS_DIR / fname, 'wb') as dst:
                    dst.write(src.read())
                count += 1
    print(f"  Extracted {count} XML annotations")
    return count


# ============================================================
# STEP 2: Match + Quality Filter
# ============================================================
def step2_match_and_filter():
    print("\n" + "=" * 60)
    print("  STEP 2: Match annotations + Quality filter")
    print("=" * 60)

    image_lookup = {}
    for ext in ["*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.png", "*.PNG"]:
        for p in SAMPLE_DIR.glob(ext):
            image_lookup[p.stem.lower()] = p
    print(f"  {len(image_lookup)} images in sample dir")

    matched = []
    class_counts = Counter()

    for xml_path in sorted(ANNOTATIONS_DIR.glob("*.xml")):
        stem = xml_path.stem.lower()
        if stem not in image_lookup:
            continue
        tree = ET.parse(str(xml_path))
        root = tree.getroot()
        classes = [obj.find("name").text for obj in root.findall("object")
                   if obj.find("name") is not None and obj.find("name").text in CLASS_MAP]
        if not classes:
            continue

        img = cv2.imread(str(image_lookup[stem]))
        if img is None:
            continue
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Quality checks
        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
        brightness = float(np.mean(gray))
        if blur < BLUR_THRESHOLD or brightness < BRIGHTNESS_MIN or brightness > BRIGHTNESS_MAX:
            print(f"    [SKIP] {xml_path.stem} blur={blur:.0f} bright={brightness:.0f}")
            continue

        matched.append({
            "image_path": image_lookup[stem],
            "xml_path": xml_path,
            "classes": classes
        })
        for c in classes:
            class_counts[c] += 1

    print(f"  Matched & passed: {len(matched)}")
    for c, n in class_counts.items():
        print(f"    {c}: {n}")
    return matched


# ============================================================
# VOC XML → YOLO conversion
# ============================================================
def parse_voc_xml(xml_path, img_w, img_h):
    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    yolo_lines = []
    bboxes = []
    class_ids = []

    for obj in root.findall("object"):
        name = obj.find("name").text
        if name not in CLASS_MAP:
            continue
        cid = CLASS_MAP[name]
        bb = obj.find("bndbox")
        x1 = max(0, int(float(bb.find("xmin").text)))
        y1 = max(0, int(float(bb.find("ymin").text)))
        x2 = min(img_w, int(float(bb.find("xmax").text)))
        y2 = min(img_h, int(float(bb.find("ymax").text)))

        xc = ((x1 + x2) / 2.0) / img_w
        yc = ((y1 + y2) / 2.0) / img_h
        bw = (x2 - x1) / img_w
        bh = (y2 - y1) / img_h
        if bw <= 0 or bh <= 0:
            continue

        yolo_lines.append(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
        bboxes.append([x1, y1, x2, y2])
        class_ids.append(cid)

    return yolo_lines, bboxes, class_ids


# ============================================================
# STEP 3: Build dataset with copy-paste + targeted augmentation
# ============================================================
def step3_build_dataset(matched):
    print("\n" + "=" * 60)
    print("  STEP 3: Building dataset (copy-paste + targeted aug)")
    print("=" * 60)

    for split in ["train", "val"]:
        (DATASET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Stratified split
    supporting = [it for it in matched if "supporting_tower" in it["classes"]]
    monopole = [it for it in matched if "monopole_tower" in it["classes"] and it not in supporting]
    random.shuffle(supporting)
    random.shuffle(monopole)

    s_val = max(2, int(len(supporting) * VAL_SPLIT))
    m_val = max(2, int(len(monopole) * VAL_SPLIT))

    val_items = supporting[:s_val] + monopole[:m_val]
    train_items = supporting[s_val:] + monopole[m_val:]
    random.shuffle(val_items)
    random.shuffle(train_items)

    print(f"  Train: {len(train_items)} | Val: {len(val_items)}")

    # Augmentation pipelines
    mono_aug = A.Compose([
        A.Rotate(limit=20, border_mode=cv2.BORDER_REFLECT_101, p=0.7),
        A.RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.35, p=0.9),
        A.RandomGamma(gamma_limit=(60, 160), p=0.6),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=30, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.OneOf([
            A.MotionBlur(blur_limit=5, p=1),
            A.GaussianBlur(blur_limit=(3, 5), p=1),
        ], p=0.2),
        A.RandomScale(scale_limit=0.2, p=0.3),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_ids'], min_visibility=0.3))

    supp_aug = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, border_mode=cv2.BORDER_REFLECT_101, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
        A.RandomScale(scale_limit=0.15, p=0.3),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_ids'], min_visibility=0.3))

    stats = Counter()

    def save_image_label(img, yolo_lines, split, stem):
        """Save image + label to dataset."""
        h, w = img.shape[:2]
        if max(h, w) > 1600:
            sc = 1600 / max(h, w)
            img = cv2.resize(img, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(DATASET_DIR / "images" / split / f"{stem}.jpg"), img)
        with open(DATASET_DIR / "labels" / split / f"{stem}.txt", "w") as f:
            f.write("\n".join(yolo_lines))

    def augment_once(img, bboxes, class_ids, transform):
        """Apply augmentation, return (aug_img, aug_yolo_lines) or None."""
        h, w = img.shape[:2]
        if max(h, w) > 1600:
            sc = 1600 / max(h, w)
            img = cv2.resize(img, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA)
            bboxes = [[int(b[0]*sc), int(b[1]*sc), int(b[2]*sc), int(b[3]*sc)] for b in bboxes]
            h, w = img.shape[:2]

        clamped = []
        cids = []
        for bb, cid in zip(bboxes, class_ids):
            x1 = max(0, min(bb[0], w-1))
            y1 = max(0, min(bb[1], h-1))
            x2 = max(x1+1, min(bb[2], w))
            y2 = max(y1+1, min(bb[3], h))
            if x2 - x1 > 5 and y2 - y1 > 5:
                clamped.append([x1, y1, x2, y2])
                cids.append(cid)
        if not clamped:
            return None

        try:
            t = transform(image=img, bboxes=clamped, class_ids=cids)
            if not t['bboxes']:
                return None
            ah, aw = t['image'].shape[:2]
            lines = []
            for bb, cid in zip(t['bboxes'], t['class_ids']):
                xc = ((bb[0]+bb[2])/2)/aw
                yc = ((bb[1]+bb[3])/2)/ah
                bw = (bb[2]-bb[0])/aw
                bh = (bb[3]-bb[1])/ah
                if 0 <= xc <= 1 and 0 <= yc <= 1 and bw > 0 and bh > 0:
                    lines.append(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
            return (t['image'], lines) if lines else None
        except Exception:
            return None

    # --- Process VAL set (no augmentation) ---
    print("  Processing validation set...")
    for item in val_items:
        img = cv2.imread(str(item["image_path"]))
        if img is None: continue
        oh, ow = img.shape[:2]
        yolo_lines, _, _ = parse_voc_xml(item["xml_path"], ow, oh)
        if not yolo_lines: continue
        save_image_label(img, yolo_lines, "val", item["image_path"].stem)
        stats["val"] += 1

    # --- Process TRAIN set with targeted augmentation ---
    print("  Processing training set with targeted augmentation...")
    monopole_crops = []  # for copy-paste
    non_monopole_items = []  # backgrounds for pasting

    for item in train_items:
        img = cv2.imread(str(item["image_path"]))
        if img is None: continue
        oh, ow = img.shape[:2]
        yolo_lines, bboxes, class_ids = parse_voc_xml(item["xml_path"], ow, oh)
        if not yolo_lines: continue

        stem = item["image_path"].stem
        has_monopole = 1 in class_ids

        # Save original
        save_image_label(img, yolo_lines, "train", stem)
        stats["train_orig"] += 1

        if has_monopole:
            # Collect monopole crops for copy-paste
            for bb, cid in zip(bboxes, class_ids):
                if cid == 1:
                    x1, y1, x2, y2 = bb
                    crop = img[y1:y2, x1:x2].copy()
                    if crop.size > 0 and crop.shape[0] > 10 and crop.shape[1] > 10:
                        monopole_crops.append(crop)

            # Heavy augmentation for monopole (8 copies)
            for i in range(MONOPOLE_AUG_COPIES):
                result = augment_once(img, bboxes, class_ids, mono_aug)
                if result:
                    save_image_label(result[0], result[1], "train", f"{stem}_monoaug{i}")
                    stats["train_mono_aug"] += 1
        else:
            non_monopole_items.append(item)
            # Light augmentation for supporting (2 copies)
            for i in range(SUPPORTING_AUG_COPIES):
                result = augment_once(img, bboxes, class_ids, supp_aug)
                if result:
                    save_image_label(result[0], result[1], "train", f"{stem}_suppaug{i}")
                    stats["train_supp_aug"] += 1

    # --- COPY-PASTE: paste monopole crops onto different backgrounds ---
    print(f"  Copy-paste: {len(monopole_crops)} monopole crops x {COPY_PASTE_PER_MONOPOLE} backgrounds...")

    bg_images = []
    for item in non_monopole_items:
        bg = cv2.imread(str(item["image_path"]))
        if bg is not None:
            bg_images.append(bg)

    if bg_images and monopole_crops:
        for crop_idx, crop in enumerate(monopole_crops):
            ch, cw = crop.shape[:2]
            targets = random.sample(bg_images, min(COPY_PASTE_PER_MONOPOLE, len(bg_images)))

            for bg_idx, bg in enumerate(targets):
                bh, bw = bg.shape[:2]

                # Scale crop to 30-55% of bg height
                scale = random.uniform(0.3, 0.55) * bh / ch
                new_w = max(1, int(cw * scale))
                new_h = max(1, int(ch * scale))
                if new_w >= bw or new_h >= bh:
                    continue

                resized = cv2.resize(crop, (new_w, new_h))

                # Random position
                px = random.randint(0, bw - new_w)
                py = random.randint(0, bh - new_h)

                # Alpha blending at edges for more natural look
                composite = bg.copy()
                # Create soft-edge mask
                mask = np.ones((new_h, new_w), dtype=np.float32)
                border = max(3, min(new_w, new_h) // 8)
                for b in range(border):
                    alpha = b / border
                    mask[b, :] *= alpha
                    mask[new_h-1-b, :] *= alpha
                    mask[:, b] *= alpha
                    mask[:, new_w-1-b] *= alpha
                mask_3c = np.stack([mask]*3, axis=-1)

                roi = composite[py:py+new_h, px:px+new_w].astype(np.float32)
                pasted = resized.astype(np.float32) * mask_3c + roi * (1 - mask_3c)
                composite[py:py+new_h, px:px+new_w] = pasted.astype(np.uint8)

                # YOLO label for pasted monopole
                xc = (px + new_w/2) / bw
                yc = (py + new_h/2) / bh
                w_norm = new_w / bw
                h_norm = new_h / bh
                yolo_line = f"1 {xc:.6f} {yc:.6f} {w_norm:.6f} {h_norm:.6f}"

                stem = f"copypaste_mono_{crop_idx}_{bg_idx}"
                save_image_label(composite, [yolo_line], "train", stem)
                stats["train_copypaste"] += 1

    # Create data.yaml
    with open(DATASET_DIR / "data.yaml", "w") as f:
        f.write(f"path: {DATASET_DIR}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write(f"\nnc: {len(CLASS_NAMES)}\n")
        f.write(f"names: {CLASS_NAMES}\n")

    print(f"\n  Dataset built at: {DATASET_DIR}")
    print(f"  Val: {stats['val']}")
    print(f"  Train originals: {stats['train_orig']}")
    print(f"  Train monopole augmented: {stats['train_mono_aug']}")
    print(f"  Train supporting augmented: {stats['train_supp_aug']}")
    print(f"  Train copy-paste monopole: {stats['train_copypaste']}")
    total = stats['train_orig'] + stats['train_mono_aug'] + stats['train_supp_aug'] + stats['train_copypaste']
    print(f"  TOTAL TRAIN: {total}")

    # Count class distribution
    lbl_dir = DATASET_DIR / "labels" / "train"
    cls_count = Counter()
    for lbl_file in lbl_dir.glob("*.txt"):
        with open(lbl_file) as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    cls_count[CLASS_NAMES[int(float(parts[0]))]] += 1
    print(f"  Final class distribution: {dict(cls_count)}")


# ============================================================
# STEP 4: Train YOLOv8 at imgsz=960
# ============================================================
def step4_train():
    print("\n" + "=" * 60)
    print(f"  STEP 4: Training YOLOv8n @ imgsz={IMGSZ} for {EPOCHS} epochs")
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

    # Check if batch 8 at 960px fits in 4GB VRAM, else use 4
    batch = BATCH_SIZE
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_mem / (1024**3)
        if vram < 5 and IMGSZ > 640:
            batch = 4
            print(f"  Reduced batch to {batch} for {IMGSZ}px on {vram:.1f}GB VRAM")

    results = model.train(
        data=data_yaml,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=batch,
        patience=0,        # train ALL epochs
        project=str(PROJECT_ROOT / "runs"),
        name="tower_final",
        device=device,
        exist_ok=True,
        agnostic_nms=True,
        iou=0.3,

        # Optimizer
        optimizer="AdamW",
        lr0=0.01,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=5,

        # Augmentation (YOLO built-in, on top of our Albumentations)
        augment=True,
        mosaic=0.8,
        mixup=0.15,
        flipud=0.3,
        fliplr=0.5,

        plots=True,
        save=True,
        save_period=10,
        val=True,
        workers=0,
        seed=42,
    )

    # Find best weights
    best_pt = PROJECT_ROOT / "runs" / "tower_final" / "weights" / "best.pt"
    if not best_pt.exists():
        candidates = list((PROJECT_ROOT / "runs").rglob("best.pt"))
        if candidates:
            best_pt = candidates[0]

    print(f"\n  Best weights: {best_pt}")

    # Evaluate
    eval_model = YOLO(str(best_pt))
    metrics = eval_model.val(data=data_yaml, split="val", plots=True)

    summary = {
        "dataset": "Final LabelImg + CopyPaste + Targeted Aug",
        "model": "YOLOv8n",
        "imgsz": IMGSZ,
        "epochs": EPOCHS,
        "conf_threshold": CONF_THRESHOLD,
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

    # Deploy
    shutil.copy2(str(best_pt), str(PROJECT_ROOT / "best_tower_model.pt"))
    (PROJECT_ROOT / "runs" / "tower_roboflow" / "weights").mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(best_pt), str(PROJECT_ROOT / "runs" / "tower_roboflow" / "weights" / "best.pt"))

    # Generate graphs
    generate_graphs(PROJECT_ROOT / "runs" / "tower_final", PROJECT_ROOT / "training_graphs")

    print("\n" + "=" * 60)
    print("  MONOPOLE RECALL FIX COMPLETE!")
    print("=" * 60)


def generate_graphs(run_dir, output_dir):
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
            print("  results.csv not found")
            return

    data = {}
    with open(csv_file, "r") as f:
        reader = csv.reader(f)
        header = [c.strip() for c in next(reader)]
        for col in header: data[col] = []
        for row in reader:
            for col, val in zip(header, row):
                try: data[col].append(float(val.strip()))
                except: data[col].append(0.0)

    epochs = data.get("epoch", list(range(len(next(iter(data.values()))))))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=200)
    fig.suptitle(f"Tower Detection - Final Model (imgsz={IMGSZ})", fontsize=16, fontweight="bold")
    for ax in axes.flat: ax.grid(True, ls="--", alpha=0.5)

    if "train/box_loss" in data:
        axes[0,0].plot(epochs, data["train/box_loss"], label="Train", color="#2563eb", lw=2)
        if "val/box_loss" in data:
            axes[0,0].plot(epochs, data["val/box_loss"], label="Val", color="#3b82f6", ls="--", lw=2)
        axes[0,0].set_title("Box Loss"); axes[0,0].legend()

    if "train/cls_loss" in data:
        axes[0,1].plot(epochs, data["train/cls_loss"], label="Train", color="#ea580c", lw=2)
        if "val/cls_loss" in data:
            axes[0,1].plot(epochs, data["val/cls_loss"], label="Val", color="#f97316", ls="--", lw=2)
        axes[0,1].set_title("Cls Loss"); axes[0,1].legend()

    if "metrics/mAP50(B)" in data:
        axes[1,0].plot(epochs, data["metrics/mAP50(B)"], label="mAP50", color="#7c3aed", lw=2.5)
        if "metrics/mAP50-95(B)" in data:
            axes[1,0].plot(epochs, data["metrics/mAP50-95(B)"], label="mAP50-95", color="#f59e0b", ls="--", lw=2)
        axes[1,0].set_title("mAP"); axes[1,0].set_ylim(0, 1.02); axes[1,0].legend()

    if "metrics/precision(B)" in data:
        axes[1,1].plot(epochs, data["metrics/precision(B)"], label="Precision", color="#0284c7", lw=2)
        if "metrics/recall(B)" in data:
            axes[1,1].plot(epochs, data["metrics/recall(B)"], label="Recall", color="#10b981", lw=2)
        axes[1,1].set_title("Precision & Recall"); axes[1,1].set_ylim(0, 1.02); axes[1,1].legend()

    plt.tight_layout(); plt.savefig(out_path / "training_overview.png"); plt.close()

    for f in ["confusion_matrix.png", "confusion_matrix_normalized.png",
              "BoxPR_curve.png", "BoxF1_curve.png", "results.png", "val_batch0_pred.jpg"]:
        src = run_path / f
        if src.exists(): shutil.copy2(src, out_path / f)
    print("  Graphs saved!")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  MONOPOLE RECALL FIX PIPELINE")
    print("  Copy-Paste + Targeted Aug + imgsz=960")
    print("  Using ONLY 124 provided images")
    print("=" * 60)

    step1_extract()
    matched = step2_match_and_filter()
    if len(matched) < 5:
        print(f"ERROR: Only {len(matched)} images passed. Need more.")
        sys.exit(1)

    step3_build_dataset(matched)
    step4_train()
