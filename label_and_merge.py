"""
label_and_merge.py — Auto-label ALL unlabeled datasets + merge with Roboflow annotations
into one unified YOLO training dataset for maximum accuracy.

Data Sources:
  1. Roboflow annotated (already labeled): 251 train + 24 val + 12 test
  2. Supporting (2): 63 images -> label as class 1 (supporting_tower)
  3. Mono: 27 images -> label as class 0 (monopole_tower)
  4. Sample: 129 images -> classify using trained model, then label

Output: dataset_merged/ with train/val/test splits and data.yaml
"""

import os
import sys
import shutil
import random
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    print("Installing Pillow...")
    os.system("pip install pillow")
    from PIL import Image

try:
    import cv2
except ImportError:
    print("Installing opencv-python...")
    os.system("pip install opencv-python")
    import cv2

# ---------- CONFIG ----------
ROBOFLOW_DIR = Path(r"C:\Users\sujay\Downloads\annotaate.v1i.yolov11")
SUPPORTING_DIR = Path(r"C:\Users\sujay\Downloads\supporting (2)\supporting\supporting")
MONO_DIR = Path(r"C:\Users\sujay\Downloads\Mono")
SAMPLE_DIR = Path(r"C:\Users\sujay\Downloads\sample\sample")

OUTPUT_DIR = Path("dataset_merged")
CLASSES = ["monopole_tower", "supporting_tower"]  # 0=monopole, 1=supporting (matches Roboflow)
VAL_SPLIT = 0.15
TEST_SPLIT = 0.10
SEED = 42

# Minimum dimension to keep
MIN_DIM = 64

random.seed(SEED)

def ensure_dirs():
    for split in ["train", "val", "test"]:
        (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

def convert_to_jpg(src_path, dst_path):
    """Convert any image format (webp, png, bmp, etc.) to JPG for consistency."""
    try:
        img = Image.open(src_path)
        if img.mode in ("RGBA", "P", "LA"):
            img = img.convert("RGB")
        img.save(dst_path, "JPEG", quality=95)
        return True
    except Exception as e:
        print(f"  [WARN] Cannot convert {src_path.name}: {e}")
        return False

def get_image_dimensions(img_path):
    """Get image width and height."""
    try:
        with Image.open(img_path) as im:
            return im.size  # (width, height)
    except:
        return None

def create_fullimage_label(class_id, margin=0.02):
    """
    Create a YOLO label that covers nearly the full image.
    Uses slight margin so the box isn't exactly edge-to-edge.
    """
    cx = 0.5
    cy = 0.5
    w = 1.0 - 2 * margin
    h = 1.0 - 2 * margin
    return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"

def copy_roboflow_data():
    """Copy already-annotated Roboflow data into the merged dataset."""
    total = 0
    for split_src, split_dst in [("train", "train"), ("valid", "val"), ("test", "test")]:
        img_src = ROBOFLOW_DIR / split_src / "images"
        lbl_src = ROBOFLOW_DIR / split_src / "labels"
        
        if not img_src.exists():
            print(f"  [SKIP] Roboflow {split_src} not found at {img_src}")
            continue
            
        img_dst = OUTPUT_DIR / "images" / split_dst
        lbl_dst = OUTPUT_DIR / "labels" / split_dst
        
        count = 0
        for img_file in img_src.iterdir():
            if img_file.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
                # Copy image
                dst_img = img_dst / img_file.name
                shutil.copy2(img_file, dst_img)
                
                # Copy matching label
                lbl_file = lbl_src / f"{img_file.stem}.txt"
                if lbl_file.exists():
                    shutil.copy2(lbl_file, lbl_dst / lbl_file.name)
                    count += 1
                    
        print(f"  Roboflow {split_src} -> {split_dst}: {count} labeled images copied")
        total += count
    return total

def label_folder_as_class(folder, class_id, class_name):
    """Label all images in a folder as a single class with full-image bounding box."""
    if not folder.exists():
        print(f"  [SKIP] {folder} does not exist")
        return []
    
    labeled = []
    for img_file in sorted(folder.iterdir()):
        if img_file.suffix.lower() not in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
            continue
            
        # Check dimensions
        dims = get_image_dimensions(img_file)
        if dims is None or min(dims) < MIN_DIM:
            print(f"  [SKIP] {img_file.name} — too small or unreadable")
            continue
        
        # Generate a unique safe filename
        safe_name = img_file.stem.replace(" ", "_").replace("(", "").replace(")", "")
        safe_name = f"{class_name}_{safe_name}.jpg"
        
        labeled.append({
            "src": img_file,
            "dst_name": safe_name,
            "label": create_fullimage_label(class_id),
            "class_id": class_id,
            "class_name": class_name
        })
    
    print(f"  {folder.name}: {len(labeled)} images labeled as {class_name} (class {class_id})")
    return labeled

def classify_sample_images():
    """Use the trained model to classify the sample images, then label them."""
    if not SAMPLE_DIR.exists():
        print("  [SKIP] Sample directory not found")
        return []
    
    # Try to load the trained model for classification
    model = None
    model_paths = [
        Path("runs/tower_roboflow/weights/best.pt"),
        Path("runs/detect/runs/tower_roboflow/weights/best.pt"),
        Path("runs/detect/runs/tower_detect/weights/best.pt"),
    ]
    for mp in model_paths:
        if mp.exists():
            try:
                from ultralytics import YOLO
                model = YOLO(str(mp))
                print(f"  Model loaded from {mp} for classifying samples")
                break
            except Exception as e:
                print(f"  [WARN] Failed to load {mp}: {e}")
    
    labeled = []
    for img_file in sorted(SAMPLE_DIR.iterdir()):
        if img_file.suffix.lower() not in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
            continue
            
        dims = get_image_dimensions(img_file)
        if dims is None or min(dims) < MIN_DIM:
            continue
        
        class_id = None
        class_name = None
        
        if model is not None:
            try:
                image = cv2.imread(str(img_file))
                if image is not None:
                    results = model(image, conf=0.05, verbose=False)[0]
                    if len(results.boxes) > 0:
                        # Use highest confidence detection
                        best_idx = results.boxes.conf.argmax()
                        cls_id = int(results.boxes.cls[best_idx])
                        cls_raw = model.names.get(cls_id, "")
                        
                        if "support" in cls_raw.lower() or "lattice" in cls_raw.lower():
                            class_id = 1
                            class_name = "supporting_tower"
                        elif "mono" in cls_raw.lower() or "pole" in cls_raw.lower():
                            class_id = 0
                            class_name = "monopole_tower"
                        else:
                            class_id = cls_id if cls_id < 2 else 1
                            class_name = CLASSES[class_id]
            except Exception as e:
                pass
        
        # Fallback: structural analysis based on file size and edge complexity
        if class_id is None:
            try:
                image = cv2.imread(str(img_file))
                if image is not None:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                    import numpy as np
                    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
                    sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
                    magnitude = np.sqrt(sobelx**2 + sobely**2)
                    angles = np.rad2deg(np.arctan2(np.abs(sobely), np.abs(sobelx)))
                    strong = magnitude > np.percentile(magnitude, 75)
                    sel_angles = angles[strong]
                    if len(sel_angles) > 0:
                        diag = np.mean((sel_angles >= 25) & (sel_angles <= 65))
                        if diag > 0.36:
                            class_id = 1
                            class_name = "supporting_tower"
                        else:
                            class_id = 0
                            class_name = "monopole_tower"
            except:
                pass
        
        if class_id is None:
            # Default to supporting (more common in dataset)
            class_id = 1
            class_name = "supporting_tower"
        
        safe_name = img_file.stem.replace(" ", "_").replace("(", "").replace(")", "")
        safe_name = f"sample_{safe_name}.jpg"
        
        labeled.append({
            "src": img_file,
            "dst_name": safe_name,
            "label": create_fullimage_label(class_id),
            "class_id": class_id,
            "class_name": class_name
        })
    
    mono_count = sum(1 for x in labeled if x["class_id"] == 0)
    supp_count = sum(1 for x in labeled if x["class_id"] == 1)
    print(f"  Sample: {len(labeled)} images classified (monopole={mono_count}, supporting={supp_count})")
    return labeled

def distribute_and_write(items):
    """Shuffle and split items into train/val/test, convert + write images and labels."""
    random.shuffle(items)
    n = len(items)
    n_test = max(int(n * TEST_SPLIT), 1)
    n_val = max(int(n * VAL_SPLIT), 1)
    n_train = n - n_val - n_test
    
    splits = {
        "train": items[:n_train],
        "val": items[n_train:n_train + n_val],
        "test": items[n_train + n_val:],
    }
    
    total_written = 0
    for split_name, split_items in splits.items():
        img_dir = OUTPUT_DIR / "images" / split_name
        lbl_dir = OUTPUT_DIR / "labels" / split_name
        
        written = 0
        for item in split_items:
            dst_img = img_dir / item["dst_name"]
            dst_lbl = lbl_dir / f"{Path(item['dst_name']).stem}.txt"
            
            # Convert image to JPG
            if item["src"].suffix.lower() in [".jpg", ".jpeg"]:
                try:
                    shutil.copy2(item["src"], dst_img)
                except Exception:
                    if not convert_to_jpg(item["src"], dst_img):
                        continue
            else:
                if not convert_to_jpg(item["src"], dst_img):
                    continue
            
            # Write label
            with open(dst_lbl, "w") as f:
                f.write(item["label"] + "\n")
            
            written += 1
        
        print(f"  -> {split_name}: {written} new images written")
        total_written += written
    
    return total_written

def write_data_yaml():
    """Write data.yaml config for YOLO training."""
    abs_path = OUTPUT_DIR.resolve()
    yaml_content = f"""path: {abs_path}
train: images/train
val: images/val
test: images/test

nc: {len(CLASSES)}
names: {CLASSES}
"""
    yaml_path = OUTPUT_DIR / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"\n  data.yaml written to {yaml_path}")
    return yaml_path

def count_final_stats():
    """Count and report final dataset statistics."""
    print("\n" + "=" * 60)
    print("    FINAL MERGED DATASET STATISTICS")
    print("=" * 60)
    
    for split in ["train", "val", "test"]:
        img_dir = OUTPUT_DIR / "images" / split
        lbl_dir = OUTPUT_DIR / "labels" / split
        n_imgs = len(list(img_dir.glob("*"))) if img_dir.exists() else 0
        n_lbls = len(list(lbl_dir.glob("*.txt"))) if lbl_dir.exists() else 0
        
        # Count per class
        mono = 0
        supp = 0
        if lbl_dir.exists():
            for lbl in lbl_dir.glob("*.txt"):
                content = lbl.read_text().strip()
                for line in content.split("\n"):
                    if line.startswith("0 "):
                        mono += 1
                    elif line.startswith("1 "):
                        supp += 1
        
        print(f"  {split:6s}: {n_imgs:4d} images, {n_lbls:4d} labels  "
              f"[monopole={mono}, supporting={supp}]")
    
    print("=" * 60)

def main():
    print("=" * 60)
    print("  AUTO-LABEL & MERGE ALL TOWER DATASETS")
    print("=" * 60)
    
    # Clean output directory
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    ensure_dirs()
    
    # 1. Copy Roboflow annotated data (already has proper bounding boxes)
    print("\n[1/4] Copying Roboflow annotated dataset...")
    roboflow_count = copy_roboflow_data()
    
    # 2. Label Supporting (2) folder as class 1
    print("\n[2/4] Labeling Supporting Tower images...")
    supporting_items = label_folder_as_class(SUPPORTING_DIR, class_id=1, class_name="supporting_tower")
    
    # 3. Label Mono folder as class 0
    print("\n[3/4] Labeling Monopole Tower images...")
    mono_items = label_folder_as_class(MONO_DIR, class_id=0, class_name="monopole_tower")
    
    # 4. Classify and label Sample images
    print("\n[4/4] Classifying & labeling Sample images...")
    sample_items = classify_sample_images()
    
    # 5. Merge new items into train/val/test splits
    print("\n[MERGE] Distributing newly labeled images across splits...")
    all_new = supporting_items + mono_items + sample_items
    if all_new:
        distribute_and_write(all_new)
    
    # 6. Write data.yaml
    write_data_yaml()
    
    # 7. Final statistics
    count_final_stats()
    
    print("\nDone! The merged dataset is ready at:", OUTPUT_DIR.resolve())
    print("Use this data.yaml path for training:")
    print(f"  {(OUTPUT_DIR / 'data.yaml').resolve()}")

if __name__ == "__main__":
    main()
