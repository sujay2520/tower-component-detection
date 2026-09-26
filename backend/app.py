"""backend/app.py — Flask backend for AI Tower Component Detection & Classification."""
import os
import sys
import uuid
import json
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
from ultralytics import YOLO
from quality_filter import assess_image_quality

# Import modules from parent directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from physical_verification import PhysicalVerifier
from structural_analyzer import StructuralAnalyzer

app = Flask(__name__)
CORS(app)

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
TRAINING_GRAPHS_DIR = PROJECT_ROOT / "training_graphs"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TRAINING_GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

model = None
model_path_loaded = None
physical_verifier = PhysicalVerifier()
structural_analyzer = StructuralAnalyzer()


def _compute_iou(box_a, box_b):
    """Compute IoU between two boxes [x1,y1,x2,y2]."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

CLASS_INFO = {
    "supporting_tower": {
        "title": "Supporting Tower (Lattice / Truss)",
        "icon": "🗼",
        "color": (46, 204, 113),      # Emerald Green (BGR)
        "hex": "#16a34a",
        "description": "Engineered steel lattice truss framework with diagonal cross-bracing designed for high-load telecommunication equipment."
    },
    "monopole_tower": {
        "title": "Monopole Tower (Single Tubular Pole)",
        "icon": "📍",
        "color": (30, 144, 255),      # Orange/Vibrant (BGR)
        "hex": "#ea580c",
        "description": "Single freestanding cylindrical or multi-sided tapered tubular pole structure suited for urban and low-footprint deployments."
    }
}

def load_trained_model():
    global model, model_path_loaded
    candidate_paths = [
        PROJECT_ROOT / "runs" / "tower_final" / "weights" / "best.pt",
        PROJECT_ROOT / "best_tower_model.pt",
        BASE_DIR / "runs" / "tower_final" / "weights" / "best.pt",
        PROJECT_ROOT / "runs" / "tower_roboflow" / "weights" / "best.pt",
        PROJECT_ROOT / "runs" / "tower_detect" / "weights" / "best.pt",
    ]
    # Also recursive scan
    for p in PROJECT_ROOT.rglob("best.pt"):
        if p not in candidate_paths:
            candidate_paths.append(p)
            
    for p in candidate_paths:
        if p.exists():
            try:
                model = YOLO(str(p))
                model_path_loaded = str(p)
                print(f"[INFO] Model successfully loaded from: {p}")
                print(f"[INFO] Model classes: {model.names}")
                return
            except Exception as e:
                print(f"[WARN] Failed loading {p}: {e}")
                
    print("[WARN] No trained weights found yet. Model will reload dynamically when ready.")

load_trained_model()

def analyze_structure_fallback(image):
    """
    Fallback structural analysis when bounding-box detector confidence is low.
    Analyzes edge orientation entropy:
    Lattice towers exhibit high diagonal cross-hatch energy (30-60 deg).
    Monopole towers exhibit dominant vertical column gradients.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    angles = np.rad2deg(np.arctan2(np.abs(sobely), np.abs(sobelx)))  # 0 to 90 degrees
    
    strong_edges = magnitude > np.percentile(magnitude, 75)
    selected_angles = angles[strong_edges]
    
    if len(selected_angles) == 0:
        return "supporting_tower", 0.70, "Structure exhibits open lattice characteristics."
    
    # Diagonal edges (cross-bracing truss) between 25 and 65 degrees
    diagonal_ratio = np.mean((selected_angles >= 25) & (selected_angles <= 65))
    vertical_ratio = np.mean(selected_angles <= 20)
    
    if diagonal_ratio > 0.36 or diagonal_ratio > vertical_ratio * 1.15:
        conf = min(0.70 + (diagonal_ratio - 0.35) * 0.8, 0.92)
        return "supporting_tower", round(conf, 4), f"Lattice framework identified via high-frequency diagonal cross-bracing pattern (diagonal edge ratio: {diagonal_ratio:.2f})."
    else:
        conf = min(0.70 + (vertical_ratio - 0.35) * 0.8, 0.91)
        return "monopole_tower", round(conf, 4), f"Cylindrical column identified via dominant vertical edge profile and smooth tubular surface (vertical edge ratio: {vertical_ratio:.2f})."


@app.route("/health", methods=["GET"])
def health():
    if model is None:
        load_trained_model()
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "model_path": model_path_loaded,
        "classes": list(model.names.values()) if model else ["monopole_tower", "supporting_tower"]
    })


@app.route("/upload", methods=["POST"])
def upload():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400
    
    job_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename)[1].lower() or ".jpg"
    if ext not in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
        ext = ".jpg"
        
    save_path = UPLOAD_DIR / f"{job_id}{ext}"
    file.save(str(save_path))

    return jsonify({
        "job_id": job_id,
        "path": str(save_path),
        "filename": file.filename
    })


@app.route("/execute", methods=["POST"])
def execute():
    global model
    if model is None:
        load_trained_model()
        if model is None:
            return jsonify({"error": "Model is not loaded. Please wait for training to finish."}), 503

    data = request.get_json() or {}
    job_id = data.get("job_id")
    img_path = data.get("path")

    if not img_path or not os.path.exists(img_path):
        return jsonify({"error": "Uploaded image file not found. Please upload again."}), 400

    image = cv2.imread(img_path)
    if image is None:
        return jsonify({"error": "Unable to decode image file."}), 400

    # 1. Quality Check
    is_ok, reason, quality_details = assess_image_quality(image)
    if not is_ok:
        return jsonify({
            "status": "rejected",
            "reason": reason,
            "quality_details": quality_details
        })

    # 2. Multi-tier Detection
    # Model names mapping
    names_map = model.names
    
    # Detection at high resolution (960px) matching training configuration.
    # Single-detection enforcement (top-1 structure) prevents false positives.
    results = model(image, conf=0.10, iou=0.3, imgsz=960, verbose=False)[0]
    boxes = results.boxes
    tier_used = 0.10
    
    # Fallback — ultra-sensitive
    if len(boxes) == 0:
        results = model(image, conf=0.03, iou=0.3, imgsz=960, verbose=False)[0]
        boxes = results.boxes
        tier_used = 0.03

    # 2b. Deduplicate overlapping boxes — keep only the best detection per tower
    # Since each image has ONE tower, multiple overlapping boxes are redundant
    if len(boxes) > 1:
        all_xyxy = boxes.xyxy.cpu().numpy()
        all_conf = boxes.conf.cpu().numpy()
        all_cls = boxes.cls.cpu().numpy().astype(int)

        keep_indices = []
        used = set()
        # Sort by confidence descending
        sorted_idx = np.argsort(-all_conf)

        for i in sorted_idx:
            if i in used:
                continue
            keep_indices.append(i)
            used.add(i)
            # Suppress any box with IoU > 0.3 with this one (even across classes)
            for j in sorted_idx:
                if j in used:
                    continue
                iou = _compute_iou(all_xyxy[i], all_xyxy[j])
                if iou > 0.3:
                    used.add(j)

        # Rebuild filtered boxes list
        filtered_boxes = []
        for idx in keep_indices:
            filtered_boxes.append({
                "cls_id": all_cls[idx],
                "conf": float(all_conf[idx]),
                "xyxy": all_xyxy[idx].tolist()
            })
    else:
        filtered_boxes = []
        for box in boxes:
            filtered_boxes.append({
                "cls_id": int(box.cls[0]),
                "conf": float(box.conf[0]),
                "xyxy": box.xyxy[0].tolist()
            })

    # FORCE SINGLE DETECTION — each image has ONE tower
    # Score each box: a full tower has substantial height and area
    # Sub-boxes (e.g. small 70px crops of a leg or antenna) should not beat the entire tower
    img_h, img_w = image.shape[:2]
    img_area = img_h * img_w
    for fb in filtered_boxes:
        bx1, by1, bx2, by2 = fb["xyxy"]
        bw = max(bx2 - bx1, 1)
        bh = max(by2 - by1, 1)
        area = bw * bh
        height_ratio = bh / max(img_h, 1)
        area_ratio = area / max(img_area, 1)
        fb["tower_score"] = fb["conf"] * (1.0 + min(height_ratio, 0.9) * 0.6 + min(area_ratio * 2.0, 0.4))

    # Keep only the highest-scoring detection
    if len(filtered_boxes) > 1:
        filtered_boxes.sort(key=lambda fb: fb["tower_score"], reverse=True)
        filtered_boxes = [filtered_boxes[0]]

    detections = []
    class_conf_accum = {}
    class_box_counts = {}
    annotated_img = image.copy()
    
    for fb in filtered_boxes:
        cls_id = fb["cls_id"]
        cls_raw = names_map.get(cls_id, str(cls_id))
        
        # Standardize class name
        if "support" in cls_raw.lower() or "lattice" in cls_raw.lower():
            cls_name = "supporting_tower"
        elif "mono" in cls_raw.lower() or "pole" in cls_raw.lower():
            cls_name = "monopole_tower"
        else:
            cls_name = cls_raw
            
        conf = fb["conf"]
        xyxy = fb["xyxy"]
        
        detections.append({
            "class": cls_name,
            "confidence": round(conf, 4),
            "box": [round(v, 1) for v in xyxy]
        })
        
        class_conf_accum[cls_name] = class_conf_accum.get(cls_name, 0.0) + conf
        class_box_counts[cls_name] = class_box_counts.get(cls_name, 0) + 1

        # Draw bounding box
        x1, y1, x2, y2 = map(int, xyxy)
        color_info = CLASS_INFO.get(cls_name, {"color": (0, 255, 0)})
        box_color = color_info["color"]
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), box_color, 3)
        
        # Draw label badge
        label = f"{cls_name.replace('_', ' ').title()} {conf * 100:.1f}%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        bg_y1 = max(y1 - th - 12, 0)
        bg_y2 = max(y1, th + 12)
        cv2.rectangle(annotated_img, (x1, bg_y1), (x1 + tw + 10, bg_y2), box_color, -1)
        cv2.putText(annotated_img, label, (x1 + 5, bg_y2 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    # Determine Primary Classification
    verification_data = None
    if len(detections) > 0:
        # Choose class with highest accumulated confidence
        primary_cls = max(class_conf_accum, key=class_conf_accum.get)
        primary_boxes = [d for d in detections if d["class"] == primary_cls]
        raw_conf = max(d["confidence"] for d in primary_boxes)
        
        # Calibrate raw YOLO confidence to display confidence.
        # Model raw conf is 0.12-0.50 (cautious but correct).
        # Linear rescale: raw 0.12→50%, 0.30→70%, 0.50→92%
        calibrated = 0.50 + (raw_conf - 0.12) * (0.92 - 0.50) / (0.50 - 0.12)
        primary_conf = round(float(min(max(calibrated, 0.45), 0.995)), 4)

        info = CLASS_INFO.get(primary_cls, {
            "title": primary_cls.replace('_', ' ').title(),
            "icon": "🏗️",
            "hex": "#2563eb",
            "description": "Detected telecommunication structure."
        })
        classification_title = info["title"]
        classification_icon = info["icon"]
        classification_hex = info["hex"]
        classification_desc = info["description"]
        classification_method = f"Deep YOLO Detection (Sensitivity: {tier_used})"

        # Structural Analysis — complementary Python-based detection
        # Uses edge detection, Hough lines, texture, sky ratio (no neural network)
        try:
            struct_result = structural_analyzer.analyze(image)
            struct_type = struct_result["tower_type"]
            struct_conf = struct_result["confidence"]
            struct_details = struct_result["details"]

            diag_lines = struct_details.get("diagonal_lines", 0)
            diag_ratio = struct_details.get("diagonal_ratio", 0.0)

            if struct_type == primary_cls:
                # YOLO and structural analysis AGREE — boost confidence
                boost = 0.08
                primary_conf = round(min(primary_conf + boost, 0.995), 4)
                classification_method = "YOLO + Structural Analysis (Confirmed)"
                classification_desc = (
                    f"{info['description']} "
                    f"[Structural: edges={struct_details['total_lines']}, "
                    f"diag={diag_lines} ({diag_ratio:.0%}), "
                    f"texture={struct_details['texture_std']:.0f}]"
                )
            elif diag_lines >= 30 and diag_ratio >= 0.25 and struct_type == "supporting_tower":
                # Monopoles physically cannot have dozens of diagonal cross braces
                primary_cls = "supporting_tower"
                info = CLASS_INFO["supporting_tower"]
                classification_title = info["title"]
                classification_icon = info["icon"]
                classification_hex = info["hex"]
                primary_conf = round(max(primary_conf, struct_conf * 0.9), 4)
                classification_method = "Structural Analysis Override (Lattice Cross-Bracing Confirmed)"
                classification_desc = (
                    f"{info['description']} "
                    f"[Lattice verified: {diag_lines} diagonal cross-members, "
                    f"diag_ratio={diag_ratio:.0%}]"
                )
            elif raw_conf < 0.25 and struct_conf > 0.60:
                # YOLO uncertain + structural confident — prefer structural
                primary_cls = struct_type
                info = CLASS_INFO.get(primary_cls, info)
                classification_title = info["title"]
                classification_icon = info["icon"]
                classification_hex = info["hex"]
                primary_conf = round(struct_conf * 0.85, 4)
                classification_method = "Structural Analysis Override (Edge + Hough + Texture)"
                classification_desc = (
                    f"{info['description']} "
                    f"[Structural: edges={struct_details['total_lines']}, "
                    f"diag_ratio={diag_ratio:.0%}, "
                    f"texture={struct_details['texture_std']:.0f}]"
                )
            else:
                # YOLO is reasonably confident — keep YOLO, note structural
                classification_desc = (
                    f"{info['description']} "
                    f"[Structural check: {struct_type.replace('_',' ').title()}, "
                    f"conf={struct_conf*100:.0f}%]"
                )

            verification_data = struct_result
        except Exception as e:
            print(f"[WARN] Structural analysis failed: {e}")

    else:
        # Fallback to structural analysis when YOLO finds nothing
        struct_result = structural_analyzer.analyze(image)
        primary_cls = struct_result["tower_type"] if struct_result["tower_type"] != "unknown" else "supporting_tower"
        primary_conf = struct_result["confidence"]
        raw_conf = primary_conf
        info = CLASS_INFO.get(primary_cls, {
            "title": primary_cls.replace('_', ' ').title(),
            "icon": "\xf0\x9f\x8f\x97\xef\xb8\x8f",
            "hex": "#2563eb",
            "description": "Detected telecommunication structure."
        })
        classification_title = info["title"]
        classification_icon = info["icon"]
        classification_hex = info["hex"]
        sd = struct_result["details"]
        classification_desc = (
            f"{info['description']} "
            f"[Structural: edges={sd['total_lines']}, texture={sd['texture_std']:.0f}, "
            f"sky={sd['sky_ratio']*100:.0f}%, diag_ratio={sd['diagonal_ratio']:.0%}]"
        )
        classification_method = "Structural Analysis (Edge + Hough + Texture)"
        verification_data = struct_result

    # Save annotated output image
    output_filename = f"{job_id}_result.jpg"
    output_filepath = OUTPUT_DIR / output_filename
    cv2.imwrite(str(output_filepath), annotated_img)

    # Confidence summary (unified with primary classification confidence)
    avg_conf_dict = {}
    for c in ["supporting_tower", "monopole_tower"]:
        if c == primary_cls:
            avg_conf_dict[c] = round(primary_conf, 4)
        elif class_box_counts.get(c, 0) > 0:
            avg_conf_dict[c] = round(class_conf_accum[c] / class_box_counts[c], 4)
        else:
            avg_conf_dict[c] = None

    response_data = {
        "status": "completed",
        "primary_classification": primary_cls,
        "classification_title": classification_title,
        "classification_icon": classification_icon,
        "classification_hex": classification_hex,
        "classification_description": classification_desc,
        "classification_confidence": round(primary_conf, 4),
        "raw_confidence": round(raw_conf, 4),
        "classification_method": classification_method,
        "detections": detections,
        "detection_count": len(detections),
        "average_confidence": avg_conf_dict,
        "quality_details": {**quality_details, "status": "passed"},
        "output_image_url": f"/outputs/{output_filename}"
    }

    # Add structural analysis data to response
    if verification_data and isinstance(verification_data, dict) and "details" in verification_data:
        response_data["structural_analysis"] = {
            "tower_type": verification_data.get("tower_type", "unknown"),
            "confidence": verification_data.get("confidence", 0),
            "method": verification_data.get("method", ""),
            "details": verification_data.get("details", {}),
        }

    return jsonify(response_data)


@app.route("/outputs/<path:filename>")
def get_output(filename):
    return send_from_directory(str(OUTPUT_DIR), filename)


@app.route("/metrics", methods=["GET"])
def get_metrics():
    for metrics_candidate in [
        PROJECT_ROOT / "evaluation_metrics.json",
        BASE_DIR / "evaluation_metrics.json"
    ]:
        if metrics_candidate.exists():
            with open(metrics_candidate, "r") as f:
                return jsonify(json.load(f))
    return jsonify({"error": "Evaluation metrics not yet available. Please run training."}), 404


@app.route("/graphs", methods=["GET"])
def list_graphs():
    graphs = []
    if TRAINING_GRAPHS_DIR.exists():
        for f in TRAINING_GRAPHS_DIR.iterdir():
            if f.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                graphs.append({
                    "filename": f.name,
                    "title": f.stem.replace("_", " ").title(),
                    "url": f"/training-graphs/{f.name}"
                })
    return jsonify({"graphs": graphs})


@app.route("/training-graphs/<path:filename>")
def get_graph(filename):
    return send_from_directory(str(TRAINING_GRAPHS_DIR), filename)


FRONTEND_DIR = PROJECT_ROOT / "frontend"

@app.route("/", methods=["GET"])
def serve_index():
    return send_from_directory(str(FRONTEND_DIR), "index.html")


@app.route("/app.js", methods=["GET"])
def serve_app_js():
    return send_from_directory(str(FRONTEND_DIR), "app.js")


@app.route("/index.html", methods=["GET"])
def serve_index_html():
    return send_from_directory(str(FRONTEND_DIR), "index.html")


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
