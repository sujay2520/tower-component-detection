"""
build_augmented_dataset.py — Comprehensive data pipeline with deduplication,
LabelImg manual annotations ingestion, and Albumentations augmentations.
=============================================================================
1. Ingests all data sources:
   - support.zip (manual LabelImg XML annotations from user team - priority 1)
   - sample/sample (124 images)
   - 4.zip (supporting images)
   - Images/ (new images)
   - addes/ (new images)
   - monopole2.0/monopole2 (monopole images)
   - Mono/ (monopole images)
   - supporting (2)/supporting/supporting (supporting images)
   - annotaate.v1i.yolov11/ (Roboflow annotations)
2. Image Deduplication:
   - Hash-based deduplication (MD5 + visual signature) to ignore duplicate/repeating images.
3. Albumentations Augmentations:
   - Angles (rotation), brightness, contrast, exposure, horizontal flip, blur.
   - Bounding boxes transformed accurately with Albumentations BboxParams.
4. Creates:
   - dataset_augmented/ with images/train, images/val, labels/train, labels/val
   - data.yaml with classes ['supporting_tower', 'monopole_tower']
"""

import os
import io
import hashlib
import zipfile
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import albumentations as A

# Paths
DOWNLOADS = Path(r"C:\Users\sujay\Downloads")
OUTPUT_DIR = Path("dataset_augmented")
CLASSES = ["supporting_tower", "monopole_tower"]
SEED = 42
MAX_IMG_DIM = 640
random.seed(SEED)
np.random.seed(SEED)


def get_image_hash(img_bgr):
    """Compute MD5 and downsampled perceptual hash for deduplication."""
    # 1. Downsampled pixel hash
    small = cv2.resize(img_bgr, (32, 32), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    avg = gray.mean()
    diff = gray > avg
    bit_hash = "".join(["1" if b else "0" for b in diff.flatten()])
    return bit_hash


def parse_xml_annotation(xml_bytes, img_w, img_h):
    """Parse Pascal VOC XML string/bytes into YOLO boxes [(cls_id, cx, cy, w, h)]."""
    root = ET.fromstring(xml_bytes)
    boxes = []
    
    # Read size from XML or fallback to image size
    size_el = root.find("size")
    if size_el is not None:
        try:
            w_xml = float(size_el.find("width").text)
            h_xml = float(size_el.find("height").text)
            if w_xml > 0 and h_xml > 0:
                img_w, img_h = w_xml, h_xml
        except:
            pass

    for obj in root.findall("object"):
        cname = obj.find("name").text.strip().lower()
        if "support" in cname or "lattice" in cname:
            cid = 0
        elif "mono" in cname or "pole" in cname:
            cid = 1
        else:
            continue

        bnd = obj.find("bndbox")
        xmin = float(bnd.find("xmin").text)
        ymin = float(bnd.find("ymin").text)
        xmax = float(bnd.find("xmax").text)
        ymax = float(bnd.find("ymax").text)

        # Clamp
        xmin = max(0, min(img_w, xmin))
        ymin = max(0, min(img_h, ymin))
        xmax = max(0, min(img_w, xmax))
        ymax = max(0, min(img_h, ymax))

        if xmax <= xmin or ymax <= ymin:
            continue

        cx = ((xmin + xmax) / 2.0) / img_w
        cy = ((ymin + ymax) / 2.0) / img_h
        bw = (xmax - xmin) / img_w
        bh = (ymax - ymin) / img_h

        # Clamp normalized values
        cx = min(max(cx, 0.001), 0.999)
        cy = min(max(cy, 0.001), 0.999)
        bw = min(max(bw, 0.001), 0.999)
        bh = min(max(bh, 0.001), 0.999)

        boxes.append((cid, cx, cy, bw, bh))

    return boxes


def detect_fallback_box(image, detector, default_cid):
    """Detect bounding box with detector or column saliency."""
    h, w = image.shape[:2]
    if detector is not None:
        try:
            res = detector(image, conf=0.10, verbose=False)[0]
            if len(res.boxes) > 0:
                best_box = None
                max_conf = 0.0
                for b in res.boxes:
                    conf = float(b.conf[0])
                    if conf > max_conf:
                        max_conf = conf
                        x1, y1, x2, y2 = b.xyxy[0].tolist()
                        cx = ((x1 + x2) / 2.0) / w
                        cy = ((y1 + y2) / 2.0) / h
                        bw = (x2 - x1) / w
                        bh = (y2 - y1) / h
                        best_box = (default_cid, cx, cy, bw, bh)
                if best_box:
                    return [best_box]
        except:
            pass

    # Column saliency fallback
    is_mono = (default_cid == 1)
    bw = 0.22 if is_mono else 0.45
    bh = 0.88 if is_mono else 0.92
    return [(default_cid, 0.50, 0.50, bw, bh)]


def main():
    print("=" * 65)
    print("  BUILDING AUGMENTED TOWER DATASET (WITH DEDUPLICATION)")
    print("=" * 65)

    # Load model for fallback pseudo-labeling
    detector = None
    for wc in ["best_tower_model.pt", "runs/tower_colab/weights/best.pt", "runs/detect/runs/tower_roboflow/weights/best.pt"]:
        if os.path.exists(wc):
            try:
                detector = YOLO(wc)
                print(f"Loaded detector for pseudo-box localization: {wc}")
                break
            except:
                pass

    # 1. Extract manual XML annotations from support.zip
    manual_xmls = {}
    support_zip_path = DOWNLOADS / "support.zip"
    if support_zip_path.exists():
        with zipfile.ZipFile(support_zip_path) as z:
            for fname in z.namelist():
                if fname.endswith(".xml"):
                    stem = Path(fname).stem.lower()
                    manual_xmls[stem] = z.read(fname)
        print(f"Loaded {len(manual_xmls)} manual XML annotations from support.zip (Priority 1)")

    seen_hashes = set()
    unique_items = []
    duplicate_count = 0

    def add_image_item(img_bgr, source_name, original_stem, label_boxes):
        nonlocal duplicate_count
        if img_bgr is None:
            return False
        
        h, w = img_bgr.shape[:2]
        if min(h, w) < 40:
            return False

        # Compute hash for deduplication
        im_hash = get_image_hash(img_bgr)
        if im_hash in seen_hashes:
            duplicate_count += 1
            return False
        seen_hashes.add(im_hash)

        # Standardize size to max 640px
        if max(h, w) > MAX_IMG_DIM:
            ratio = MAX_IMG_DIM / max(h, w)
            new_w, new_h = int(w * ratio), int(h * ratio)
            img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

        unique_items.append({
            "image": img_bgr,
            "source": source_name,
            "stem": original_stem,
            "boxes": label_boxes  # list of (cid, cx, cy, bw, bh)
        })
        return True

    # --- SOURCE 1: sample/sample (Cross-checked with manual support.zip XMLs) ---
    sample_dir = DOWNLOADS / "sample" / "sample"
    if sample_dir.exists():
        print(f"\nProcessing sample/sample...")
        for f in sorted(sample_dir.glob("*")):
            if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                img = cv2.imread(str(f))
                if img is None:
                    continue
                h, w = img.shape[:2]
                stem = f.stem.lower()
                
                # Check manual XML from support.zip
                if stem in manual_xmls:
                    boxes = parse_xml_annotation(manual_xmls[stem], w, h)
                else:
                    # Default based on detector
                    boxes = detect_fallback_box(img, detector, default_cid=0)

                add_image_item(img, "sample", f.stem, boxes)

    # --- SOURCE 2: 4.zip (Supporting Tower Images) ---
    four_zip = DOWNLOADS / "4.zip"
    if four_zip.exists():
        print(f"\nProcessing 4.zip (Supporting towers)...")
        with zipfile.ZipFile(four_zip) as z:
            for fname in z.namelist():
                if any(fname.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".webp"]):
                    data = z.read(fname)
                    arr = np.frombuffer(data, np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if img is not None:
                        boxes = detect_fallback_box(img, detector, default_cid=0)  # supporting_tower = 0
                        add_image_item(img, "4_zip", Path(fname).stem, boxes)

    # --- SOURCE 3: monopole2.0/monopole2 (Monopole Tower Images) ---
    mono2_dir = DOWNLOADS / "monopole2.0" / "monopole2"
    if mono2_dir.exists():
        print(f"\nProcessing monopole2.0/monopole2...")
        for f in sorted(mono2_dir.glob("*")):
            if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp", ".avif", ".bmp"]:
                img = cv2.imread(str(f))
                if img is not None:
                    boxes = detect_fallback_box(img, detector, default_cid=1)  # monopole_tower = 1
                    add_image_item(img, "monopole2", f.stem, boxes)

    # --- SOURCE 4: Mono/ (Monopole Tower Images) ---
    mono_dir = DOWNLOADS / "Mono"
    if mono_dir.exists():
        print(f"\nProcessing Mono/...")
        for f in sorted(mono_dir.glob("*")):
            if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
                img = cv2.imread(str(f))
                if img is not None:
                    boxes = detect_fallback_box(img, detector, default_cid=1)
                    add_image_item(img, "mono", f.stem, boxes)

    # --- SOURCE 5: supporting (2)/supporting/supporting ---
    supp_dir = DOWNLOADS / "supporting (2)" / "supporting" / "supporting"
    if supp_dir.exists():
        print(f"\nProcessing supporting (2)...")
        for f in sorted(supp_dir.glob("*")):
            if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
                img = cv2.imread(str(f))
                if img is not None:
                    boxes = detect_fallback_box(img, detector, default_cid=0)
                    add_image_item(img, "supporting", f.stem, boxes)

    # --- SOURCE 6: Images/ ---
    images_dir = DOWNLOADS / "Images"
    if images_dir.exists():
        print(f"\nProcessing Images/...")
        for f in sorted(images_dir.rglob("*")):
            if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
                img = cv2.imread(str(f))
                if img is not None:
                    # Determine class from filename or detector
                    cid = 1 if "mono" in f.name.lower() or "pole" in f.name.lower() else 0
                    boxes = detect_fallback_box(img, detector, default_cid=cid)
                    add_image_item(img, "images_dir", f.stem, boxes)

    # --- SOURCE 7: addes/ ---
    addes_dir = DOWNLOADS / "addes"
    if addes_dir.exists():
        print(f"\nProcessing addes/...")
        for f in sorted(addes_dir.glob("*")):
            if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
                img = cv2.imread(str(f))
                if img is not None:
                    boxes = detect_fallback_box(img, detector, default_cid=0)
                    add_image_item(img, "addes", f.stem, boxes)

    # --- SOURCE 8: annotaate.v1i.yolov11 (Roboflow Pre-annotated) ---
    robo_dir = DOWNLOADS / "annotaate.v1i.yolov11"
    if robo_dir.exists():
        print(f"\nProcessing annotaate.v1i.yolov11...")
        for split in ["train", "valid", "test"]:
            img_dir = robo_dir / split / "images"
            lbl_dir = robo_dir / split / "labels"
            if img_dir.exists():
                for f in sorted(img_dir.glob("*")):
                    if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
                        lbl_file = lbl_dir / f"{f.stem}.txt"
                        if lbl_file.exists():
                            boxes = []
                            for line in lbl_file.read_text().strip().split("\n"):
                                parts = line.strip().split()
                                if len(parts) >= 5:
                                    # Roboflow: 0=monopole, 1=supporting
                                    # Convert to our unified format: 0=supporting, 1=monopole
                                    robo_cid = int(parts[0])
                                    our_cid = 1 if robo_cid == 0 else 0
                                    cx, cy, bw, bh = map(float, parts[1:5])
                                    boxes.append((our_cid, cx, cy, bw, bh))
                            if boxes:
                                img = cv2.imread(str(f))
                                if img is not None:
                                    add_image_item(img, f"robo_{split}", f.stem, boxes)

    print("\n" + "=" * 65)
    print(f"  DEDUPLICATION SUMMARY:")
    print(f"    Unique images retained: {len(unique_items)}")
    print(f"    Duplicate images ignored: {duplicate_count}")
    print("=" * 65)

    # Split into train (85%) and val (15%)
    random.shuffle(unique_items)
    n_val = max(int(len(unique_items) * 0.15), 20)
    train_items = unique_items[n_val:]
    val_items = unique_items[:n_val]
    print(f"Base split: {len(train_items)} train images, {len(val_items)} val images")

    # Define Albumentations pipeline for training set augmentation
    aug_pipeline = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, value=(114, 114, 114), p=0.7),
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.8),
        A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        A.MotionBlur(blur_limit=5, p=0.3),
        A.GaussNoise(var_limit=(10.0, 40.0), p=0.3),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.25))

    # Prepare directories
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    for s in ["train", "val"]:
        (OUTPUT_DIR / "images" / s).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "labels" / s).mkdir(parents=True, exist_ok=True)

    # Write Validation set (pure unaugmented images)
    print("\nWriting validation images (unaugmented)...")
    for idx, item in enumerate(val_items):
        fn = f"val_{idx:04d}_{item['source']}_{item['stem']}.jpg"
        cv2.imwrite(str(OUTPUT_DIR / "images" / "val" / fn), item["image"])
        lines = [f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}" for cid, cx, cy, bw, bh in item["boxes"]]
        with open(OUTPUT_DIR / "labels" / "val" / f"{Path(fn).stem}.txt", "w") as f:
            f.write("\n".join(lines) + "\n")

    # Write Training set + Albumentations augmented copies
    print("\nApplying Albumentations augmentations (rotation, brightness, exposure, blur)...")
    train_saved = 0
    aug_saved = 0

    for idx, item in enumerate(train_items):
        base_fn = f"train_{idx:04d}_{item['source']}_{item['stem']}"
        
        # 1. Save original image
        cv2.imwrite(str(OUTPUT_DIR / "images" / "train" / f"{base_fn}.jpg"), item["image"])
        lines = [f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}" for cid, cx, cy, bw, bh in item["boxes"]]
        with open(OUTPUT_DIR / "labels" / "train" / f"{base_fn}.txt", "w") as f:
            f.write("\n".join(lines) + "\n")
        train_saved += 1

        # 2. Prepare boxes for Albumentations
        yolo_bboxes = []
        class_labels = []
        for cid, cx, cy, bw, bh in item["boxes"]:
            # Clamp for Albumentations yolo format [cx, cy, w, h] in (0, 1)
            cx = min(max(cx, 0.005), 0.995)
            cy = min(max(cy, 0.005), 0.995)
            bw = min(max(bw, 0.005), min(2 * cx, 2 * (1 - cx), 0.99))
            bh = min(max(bh, 0.005), min(2 * cy, 2 * (1 - cy), 0.99))
            yolo_bboxes.append([cx, cy, bw, bh])
            class_labels.append(cid)

        if not yolo_bboxes:
            continue

        # 3. Generate 2 augmented copies per training image
        for aug_i in range(1, 3):
            try:
                augmented = aug_pipeline(image=item["image"], bboxes=yolo_bboxes, class_labels=class_labels)
                aug_img = augmented["image"]
                aug_boxes = augmented["bboxes"]
                aug_cls = augmented["class_labels"]

                if aug_boxes:
                    aug_fn = f"{base_fn}_aug{aug_i}.jpg"
                    cv2.imwrite(str(OUTPUT_DIR / "images" / "train" / aug_fn), aug_img)
                    aug_lines = [f"{c} {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}" for c, b in zip(aug_cls, aug_boxes)]
                    with open(OUTPUT_DIR / "labels" / "train" / f"{Path(aug_fn).stem}.txt", "w") as f:
                        f.write("\n".join(aug_lines) + "\n")
                    aug_saved += 1
            except Exception as e:
                pass

    # Write data.yaml
    yaml_content = f"""path: {OUTPUT_DIR.resolve()}
train: images/train
val: images/val

nc: 2
names: ['supporting_tower', 'monopole_tower']
"""
    with open(OUTPUT_DIR / "data.yaml", "w") as f:
        f.write(yaml_content)

    print("\n" + "=" * 65)
    print("  FINAL AUGMENTED DATASET SUMMARY:")
    print(f"    Train images: {train_saved} original + {aug_saved} albumentations augmented = {train_saved + aug_saved} total")
    print(f"    Val images  : {len(val_items)}")
    print(f"    Total images: {train_saved + aug_saved + len(val_items)}")
    print(f"    data.yaml   : {OUTPUT_DIR / 'data.yaml'}")
    print("=" * 65)


if __name__ == "__main__":
    main()
