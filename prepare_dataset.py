"""
prepare_dataset.py
Auto-labels tower images using Gemini API for classification,
then creates YOLO-format dataset with train/val split.

Requires: pip install google-genai pillow
Set GEMINI_API_KEY environment variable before running.
"""

import os
import json
import shutil
import random
import time
from pathlib import Path
from PIL import Image

# ---- CONFIG ----
RAW_IMAGES_DIR = r"C:\Users\yashw\Downloads\sample\sample"
OUTPUT_DIR = "dataset"
CLASSES = ["supporting_tower", "monopole_tower"]
VAL_SPLIT = 0.2
SEED = 42
CLASSIFICATION_CACHE = "classification_results.json"
# -----------------

random.seed(SEED)


def classify_with_gemini(image_path: str) -> str:
    """Use Gemini to classify a tower image."""
    from google import genai
    
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    
    img = Image.open(image_path)
    # Resize large images to speed up API upload (classification doesn't need full res)
    max_dim = 800
    if max(img.size) > max_dim:
        img.thumbnail((max_dim, max_dim), Image.LANCZOS)
    
    prompt = """Look at this image of a telecommunications tower. Classify it as exactly one of:

1. "supporting_tower" - A lattice tower / self-supporting tower with a triangular or square cross-section made of steel lattice/truss framework. The structure has visible cross-bracing, diagonal members, and an open framework design. Also known as a guyed tower or self-supporting lattice tower.

2. "monopole_tower" - A monopole tower consisting of a single, smooth, tapered cylindrical or conical steel pole. The pole itself is a solid/tubular structure (not lattice). Antennas and equipment are mounted on top or on arms extending from the pole.

Respond with ONLY the class name, nothing else. Either "supporting_tower" or "monopole_tower"."""
    
    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=[prompt, img]
    )
    
    result = response.text.strip().lower().replace('"', '').replace("'", "")
    
    # Fuzzy match
    if "supporting" in result or "lattice" in result or "self-supporting" in result or "guyed" in result:
        return "supporting_tower"
    elif "monopole" in result or "mono" in result:
        return "monopole_tower"
    else:
        print(f"  WARNING: unexpected classification '{result}' for {image_path}, defaulting to monopole_tower")
        return "monopole_tower"


def create_yolo_label(img_path: str, cls_id: int) -> str:
    """Create a YOLO label line with a bounding box covering the tower.
    Since each image has one main tower subject, we create a generous center box."""
    with Image.open(img_path) as im:
        w, h = im.size
    
    # Tower-centric bounding box: centered, covering ~75% of image
    # Slightly taller than wide since towers are vertical
    x_center = 0.50
    y_center = 0.48  # slightly above center (towers often extend upward)
    box_w = 0.70
    box_h = 0.85
    
    return f"{cls_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}"


def main():
    img_extensions = {".jpg", ".jpeg", ".png"}
    img_files = sorted([
        f for f in os.listdir(RAW_IMAGES_DIR)
        if Path(f).suffix.lower() in img_extensions
    ])
    print(f"Found {len(img_files)} images in {RAW_IMAGES_DIR}")
    
    # Load or create classification cache
    cache = {}
    if os.path.exists(CLASSIFICATION_CACHE):
        with open(CLASSIFICATION_CACHE) as f:
            cache = json.load(f)
        print(f"Loaded {len(cache)} cached classifications")
    
    # Classify all images
    print("\n--- Classifying images with Gemini ---")
    for i, fname in enumerate(img_files):
        if fname in cache:
            continue
        img_path = os.path.join(RAW_IMAGES_DIR, fname)
        try:
            cls_name = classify_with_gemini(img_path)
            cache[fname] = cls_name
            print(f"  [{i+1}/{len(img_files)}] {fname} -> {cls_name}")
            time.sleep(0.3)  # small delay to avoid rate limits
        except Exception as e:
            print(f"  [{i+1}/{len(img_files)}] ERROR classifying {fname}: {e}")
            cache[fname] = "monopole_tower"  # safe default
        
        # Save cache periodically
        if (i + 1) % 10 == 0:
            with open(CLASSIFICATION_CACHE, "w") as f:
                json.dump(cache, f, indent=2)
    
    # Final cache save
    with open(CLASSIFICATION_CACHE, "w") as f:
        json.dump(cache, f, indent=2)
    
    # Print distribution
    counts = {}
    for cls_name in cache.values():
        counts[cls_name] = counts.get(cls_name, 0) + 1
    print(f"\nClassification distribution: {counts}")
    
    # Shuffle and split
    labeled_files = [(f, cache[f]) for f in img_files if f in cache]
    random.shuffle(labeled_files)
    split_idx = int(len(labeled_files) * (1 - VAL_SPLIT))
    train_files = labeled_files[:split_idx]
    val_files = labeled_files[split_idx:]
    
    # Write dataset
    for split_name, files in [("train", train_files), ("val", val_files)]:
        img_out = Path(OUTPUT_DIR) / "images" / split_name
        lbl_out = Path(OUTPUT_DIR) / "labels" / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        
        for fname, cls_name in files:
            src_img = Path(RAW_IMAGES_DIR) / fname
            cls_id = CLASSES.index(cls_name)
            stem = Path(fname).stem
            
            # Copy image
            shutil.copy(src_img, img_out / fname)
            
            # Write YOLO label
            label_line = create_yolo_label(str(src_img), cls_id)
            with open(lbl_out / f"{stem}.txt", "w") as f:
                f.write(label_line)
        
        print(f"{split_name}: {len(files)} images written")
    
    # Write data.yaml
    abs_output = os.path.abspath(OUTPUT_DIR).replace("\\", "/")
    yaml_content = f"""train: {abs_output}/images/train
val: {abs_output}/images/val
nc: {len(CLASSES)}
names: {CLASSES}
"""
    with open(Path(OUTPUT_DIR) / "data.yaml", "w") as f:
        f.write(yaml_content)
    
    print(f"\nDone! Dataset written to {OUTPUT_DIR}/")
    print(f"Classification results saved to {CLASSIFICATION_CACHE}")
    print("\nIMPORTANT: Review classification_results.json and correct any misclassifications.")
    print("Then re-run this script (cached results will be used).")


if __name__ == "__main__":
    main()
