"""Diagnose monopole detection on ALL available images."""
import os, glob
from pathlib import Path
from ultralytics import YOLO
from collections import Counter

MODEL_PATH = "best_tower_model.pt"
SAMPLE_DIR = Path(r"C:\Users\sujay\Downloads\sample\sample")
VAL_DIR = Path("dataset_final/images/val")
TRAIN_DIR = Path("dataset_final/images/train")

# 1. Check val for leaks
print("=" * 60)
print("  STEP 1: Checking validation set integrity")
print("=" * 60)
val_files = list(VAL_DIR.glob("*"))
leaks = [f.name for f in val_files if "aug" in f.name.lower() or "paste" in f.name.lower() or "copy" in f.name.lower()]
print(f"Val images: {len(val_files)}")
if leaks:
    print(f"LEAK! {len(leaks)} augmented images in val: {leaks[:5]}")
else:
    print("No leaks - val is clean (originals only)")

print("\nVal files:")
for f in sorted(val_files):
    print(f"  {f.name}")

# 2. Find images NOT used in train or val (held-out test set)
print("\n" + "=" * 60)
print("  STEP 2: Finding held-out test images")
print("=" * 60)

train_stems = set()
for f in TRAIN_DIR.glob("*"):
    # Get the original stem (strip augmentation suffixes)
    stem = f.stem
    for suffix in ["_monoaug0","_monoaug1","_monoaug2","_monoaug3","_monoaug4",
                    "_monoaug5","_monoaug6","_monoaug7","_suppaug0","_suppaug1"]:
        stem = stem.replace(suffix, "")
    train_stems.add(stem.lower())

val_stems = set(f.stem.lower() for f in val_files)

used_stems = train_stems | val_stems
# Also remove copypaste stems
used_stems = {s for s in used_stems if not s.startswith("copypaste_")}

sample_images = []
for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
    sample_images.extend(SAMPLE_DIR.glob(ext))

heldout = [img for img in sample_images if img.stem.lower() not in used_stems]
print(f"Total sample images: {len(sample_images)}")
print(f"Used in train/val: {len(used_stems)}")
print(f"Held-out (unseen): {len(heldout)}")
for h in heldout[:10]:
    print(f"  {h.name}")

# 3. Run diagnostic on ALL sample images
print("\n" + "=" * 60)
print("  STEP 3: Running model on ALL sample images (conf=0.10)")
print("=" * 60)

model = YOLO(MODEL_PATH)
CONF = 0.10  # very low to see weak detections

results_summary = {"detected": 0, "missed": 0, "low_conf": 0, "high_conf": 0}
class_counts = Counter()
low_conf_images = []

for img_path in sorted(sample_images):
    preds = model.predict(source=str(img_path), conf=CONF, iou=0.3, verbose=False)[0]
    name = img_path.name
    in_val = img_path.stem.lower() in val_stems
    in_train = img_path.stem.lower() in train_stems
    split_tag = "[VAL]" if in_val else "[TRAIN]" if in_train else "[UNSEEN]"

    if len(preds.boxes) == 0:
        print(f"  {split_tag} {name} -> NO DETECTION (even at conf 0.10)")
        results_summary["missed"] += 1
        continue

    # Take best detection
    best_idx = preds.boxes.conf.argmax()
    cls_id = int(preds.boxes.cls[best_idx])
    cls_name = model.names[cls_id]
    conf = float(preds.boxes.conf[best_idx])
    class_counts[cls_name] += 1
    results_summary["detected"] += 1

    flag = ""
    if conf < 0.371:
        flag = " <-- BELOW 0.371 threshold"
        results_summary["low_conf"] += 1
        low_conf_images.append((name, cls_name, conf, split_tag))
    else:
        results_summary["high_conf"] += 1

    print(f"  {split_tag} {name} -> {cls_name} conf={conf:.3f}{flag}")

print(f"\n{'=' * 60}")
print(f"  SUMMARY")
print(f"{'=' * 60}")
print(f"  Detected: {results_summary['detected']}/{results_summary['detected']+results_summary['missed']}")
print(f"  Missed (no detection): {results_summary['missed']}")
print(f"  Above threshold (>0.371): {results_summary['high_conf']}")
print(f"  Below threshold (<0.371): {results_summary['low_conf']}")
print(f"  Class distribution: {dict(class_counts)}")

if low_conf_images:
    print(f"\n  Low-confidence detections (would be hidden in dashboard):")
    for name, cls, conf, tag in low_conf_images:
        print(f"    {tag} {name} -> {cls} conf={conf:.3f}")
