"""resize_dataset.py -- Resize all images in dataset_merged to max 1280px to prevent OOM during training."""
from pathlib import Path
from PIL import Image
import os

DATASET_DIR = Path("dataset_merged")
MAX_DIM = 1280  # max dimension on any side

def resize_image(img_path, max_dim=MAX_DIM):
    """Resize image if larger than max_dim, keeping aspect ratio."""
    try:
        with Image.open(img_path) as im:
            if im.mode in ("RGBA", "P", "LA"):
                im = im.convert("RGB")
            w, h = im.size
            if max(w, h) > max_dim:
                ratio = max_dim / max(w, h)
                new_w = int(w * ratio)
                new_h = int(h * ratio)
                im = im.resize((new_w, new_h), Image.LANCZOS)
                im.save(img_path, "JPEG", quality=92)
                return True, f"{w}x{h} -> {new_w}x{new_h}"
            elif img_path.suffix.lower() in [".webp", ".bmp", ".png"]:
                # Convert non-JPG to JPG
                new_path = img_path.with_suffix(".jpg")
                im.save(new_path, "JPEG", quality=92)
                if new_path != img_path:
                    os.remove(img_path)
                return True, f"converted to jpg"
        return False, "ok"
    except Exception as e:
        return False, str(e)

def main():
    total = 0
    resized = 0
    for split in ["train", "val", "test"]:
        img_dir = DATASET_DIR / "images" / split
        if not img_dir.exists():
            continue
        for img_file in img_dir.iterdir():
            if img_file.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
                total += 1
                changed, info = resize_image(img_file)
                if changed:
                    resized += 1
    
    print(f"Processed {total} images, resized/converted {resized}")

if __name__ == "__main__":
    main()
