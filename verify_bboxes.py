"""Verify bounding box quality of LabelImg annotations."""
import cv2
import xml.etree.ElementTree as ET
from pathlib import Path

SAMPLE_DIR = Path(r"C:\Users\sujay\Downloads\sample\sample")
ANNO_DIR = Path("annotations_extracted")

issues = []
stats = {"total": 0, "good": 0, "too_small": 0, "too_large": 0, "out_of_bounds": 0, "tiny": 0}

for xml_path in sorted(ANNO_DIR.glob("*.xml")):
    stem = xml_path.stem
    img_path = None
    for ext in ["jpg", "jpeg", "png", "JPG", "JPEG", "PNG"]:
        p = SAMPLE_DIR / f"{stem}.{ext}"
        if p.exists():
            img_path = p
            break
    if not img_path:
        continue

    img = cv2.imread(str(img_path))
    if img is None:
        continue
    ih, iw = img.shape[:2]

    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    for obj in root.findall("object"):
        name = obj.find("name").text
        bb = obj.find("bndbox")
        x1 = int(float(bb.find("xmin").text))
        y1 = int(float(bb.find("ymin").text))
        x2 = int(float(bb.find("xmax").text))
        y2 = int(float(bb.find("ymax").text))

        stats["total"] += 1
        bw = x2 - x1
        bh = y2 - y1
        area_ratio = (bw * bh) / (iw * ih)

        problems = []

        if x1 < 0 or y1 < 0 or x2 > iw or y2 > ih:
            problems.append(f"OUT_OF_BOUNDS ({x1},{y1},{x2},{y2}) img=({iw}x{ih})")
            stats["out_of_bounds"] += 1

        if area_ratio < 0.05:
            problems.append(f"TOO_SMALL area={area_ratio*100:.1f}% ({bw}x{bh})")
            stats["too_small"] += 1

        if area_ratio > 0.95:
            problems.append(f"TOO_LARGE area={area_ratio*100:.1f}% (whole image?)")
            stats["too_large"] += 1

        if bw < 20 or bh < 20:
            problems.append(f"TINY {bw}x{bh}px")
            stats["tiny"] += 1

        if problems:
            issues.append(f"  {stem} [{name}]: " + ", ".join(problems))
        else:
            stats["good"] += 1

print("=== BOUNDING BOX QUALITY REPORT ===")
total = stats["total"]
good = stats["good"]
print(f"Total annotations: {total}")
print(f"Good bboxes: {good} ({good/max(total,1)*100:.0f}%)")
print(f"Too small (<5% area): {stats['too_small']}")
print(f"Too large (>95% area): {stats['too_large']}")
print(f"Out of bounds: {stats['out_of_bounds']}")
print(f"Tiny (<20px): {stats['tiny']}")
print()
if issues:
    print("Issues found:")
    for iss in issues[:20]:
        print(iss)
    if len(issues) > 20:
        print(f"  ... and {len(issues)-20} more")
else:
    print("No issues found - all bboxes look good!")
