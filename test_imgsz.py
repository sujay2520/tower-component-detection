"""Quick diagnostic: compare confidence at imgsz=640 vs imgsz=960"""
from pathlib import Path
from ultralytics import YOLO

model = YOLO("best_tower_model.pt")
SAMPLE_DIR = Path(r"C:\Users\sujay\Downloads\sample\sample")

# Pick 10 diverse test images
test_images = sorted(SAMPLE_DIR.glob("*.jpg"))[:5] + sorted(SAMPLE_DIR.glob("*.JPG"))[:5]

print(f"{'Image':<45} {'Class':<20} {'conf@640':>10} {'conf@960':>10}  {'Δ':>6}")
print("-" * 95)

for img in test_images:
    # At 640 (wrong)
    r640 = model.predict(str(img), conf=0.05, imgsz=640, verbose=False)[0]
    # At 960 (correct)  
    r960 = model.predict(str(img), conf=0.05, imgsz=960, verbose=False)[0]
    
    c640 = float(r640.boxes.conf[0]) if len(r640.boxes) > 0 else 0
    c960 = float(r960.boxes.conf[0]) if len(r960.boxes) > 0 else 0
    
    cls640 = model.names[int(r640.boxes.cls[0])] if len(r640.boxes) > 0 else "NONE"
    cls960 = model.names[int(r960.boxes.cls[0])] if len(r960.boxes) > 0 else "NONE"
    
    delta = c960 - c640
    marker = "🔥" if delta > 0.1 else "✅" if delta > 0 else "⚠️"
    print(f"{img.name:<45} {cls960:<20} {c640:>9.3f}  {c960:>9.3f}  {delta:>+.3f} {marker}")

print()
print("If conf@960 >> conf@640, the imgsz mismatch was the bug.")
