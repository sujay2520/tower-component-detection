"""app.py — Flask backend for Tower Component Detection."""
import os
import uuid
import json
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2
from ultralytics import YOLO
from quality_filter import assess_image_quality

app = Flask(__name__)
CORS(app)

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Try to load trained model, fall back gracefully
MODEL_PATH = os.environ.get("MODEL_PATH", "../runs/tower_detect/weights/best.pt")
model = None

def load_model():
    global model
    paths_to_try = [
        MODEL_PATH,
        "../runs/tower_detect/weights/best.pt",
        "../runs/tower_detect2/weights/best.pt",
        "../runs/tower_detect3/weights/best.pt",
    ]
    # Also search recursively
    for p in Path("..").rglob("best.pt"):
        paths_to_try.append(str(p))
    
    for p in paths_to_try:
        if os.path.exists(p):
            model = YOLO(p)
            print(f"Model loaded from: {p}")
            return
    
    print("WARNING: No trained model found. /execute will return an error.")
    print("Train the model first with: python train.py")

load_model()

CLASSES = ["supporting_tower", "monopole_tower"]
CLASS_COLORS = {
    "supporting_tower": (0, 255, 0),   # green
    "monopole_tower": (255, 165, 0),    # orange (BGR)
}


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH
    })


@app.route("/upload", methods=["POST"])
def upload():
    """Receive image, save it, return job id."""
    if "image" not in request.files:
        return jsonify({"error": "no image provided"}), 400

    file = request.files["image"]
    if not file.filename:
        return jsonify({"error": "empty filename"}), 400
    
    job_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename)[1] or ".jpg"
    save_path = os.path.join(UPLOAD_DIR, f"{job_id}{ext}")
    file.save(save_path)

    return jsonify({"job_id": job_id, "path": save_path, "filename": file.filename})


@app.route("/execute", methods=["POST"])
def execute():
    """Run quality check + detection on uploaded image."""
    data = request.get_json()
    job_id = data.get("job_id")
    path = data.get("path")

    if not path or not os.path.exists(path):
        return jsonify({"error": "image not found, upload first"}), 400

    image = cv2.imread(path)
    if image is None:
        return jsonify({"error": "could not read image file"}), 400

    # --- Quality Check ---
    is_ok, reason, quality_details = assess_image_quality(image)
    if not is_ok:
        return jsonify({
            "status": "rejected",
            "reason": reason,
            "quality_details": quality_details
        })

    # --- Detection ---
    if model is None:
        return jsonify({"error": "model not loaded, train first"}), 503

    results = model(image, conf=0.25)[0]

    detections = []
    conf_sums = {c: 0.0 for c in CLASSES}
    conf_counts = {c: 0 for c in CLASSES}

    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id >= len(CLASSES):
            continue
        cls_name = CLASSES[cls_id]
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].tolist()

        detections.append({
            "class": cls_name,
            "confidence": round(conf, 4),
            "box": [round(v, 1) for v in xyxy]
        })
        conf_sums[cls_name] += conf
        conf_counts[cls_name] += 1

        # Draw box on image
        x1, y1, x2, y2 = map(int, xyxy)
        color = CLASS_COLORS.get(cls_name, (0, 255, 0))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
        label = f"{cls_name} {conf:.2f}"
        # Label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(image, (x1, max(y1 - th - 10, 0)), (x1 + tw, max(y1, th + 10)), color, -1)
        cv2.putText(image, label, (x1, max(y1 - 5, th + 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    avg_confidence = {
        c: round(conf_sums[c] / conf_counts[c], 4) if conf_counts[c] > 0 else None
        for c in CLASSES
    }

    output_filename = f"{job_id}_result.jpg"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    cv2.imwrite(output_path, image)

    return jsonify({
        "status": "completed",
        "detections": detections,
        "detection_count": len(detections),
        "average_confidence": avg_confidence,
        "quality_details": {**quality_details, "status": "passed"},
        "output_image_url": f"/outputs/{output_filename}"
    })


@app.route("/outputs/<filename>")
def get_output(filename):
    return send_from_directory(os.path.abspath(OUTPUT_DIR), filename)


@app.route("/metrics", methods=["GET"])
def get_metrics():
    """Return training evaluation metrics if available."""
    metrics_path = "../evaluation_metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            return jsonify(json.load(f))
    return jsonify({"error": "no evaluation metrics available yet"}), 404


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
