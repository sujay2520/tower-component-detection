# 🗼 AI Tower Component Detection & Quality Assessment System

An end-to-end Computer Vision system for telecommunications tower component detection and image quality assessment. Built with **YOLOv8**, **Flask**, **OpenCV**, and a responsive web dashboard.

---

## 🚀 Key Features

- **Pre-inference Image Quality Filter**:
  - **Blur Detection**: Laplacian variance thresholding (`threshold=80.0`).
  - **Exposure Assessment**: Mean pixel brightness validation (`underexposed < 40`, `overexposed > 210`).
  - **Resolution Safeguards**: Rejects low-resolution samples (< 100px).
- **YOLOv8 Custom Object Detection**:
  - Identifies `supporting_tower` (open steel lattice / truss framework with cross-bracing) and `monopole_tower` (single tapered cylindrical steel pole).
  - Pretrained transfer learning from YOLOv8n with multi-scale augmentations (mosaic, flips, rotation, HSV).
- **Interactive Web Dashboard**:
  - Drag-and-drop image upload with real-time upload progress.
  - Pre-detection quality inspection breakdown.
  - Visual bounding box overlay with per-class confidence indicators.
- **RESTful Flask Backend API**:
  - Modular API for `/upload`, `/execute`, `/health`, and `/metrics`.

---

## 🏆 Model Performance & Evaluation

Trained on 124 curated telecommunications tower samples (80/20 train/validation split):

| Metric | Score |
|---|---|
| **mAP@50** | **0.9588 (95.9%)** |
| **mAP@50-95** | **0.8358 (83.6%)** |
| **Precision** | **0.9489 (94.9%)** |
| **Recall** | **0.9455 (94.6%)** |

### Per-Class Performance
| Class | Precision | Recall | mAP@50 | mAP@50-95 |
|---|---|---|---|---|
| **Supporting Tower** | **0.984** | **0.929** | **0.983** | **0.886** |
| **Monopole Tower** | **0.914** | **0.962** | **0.935** | **0.786** |

Confusion matrices and Precision-Recall curves are available under `runs/detect/val/` and `runs/detect/runs/tower_detect/`.

---

## 📂 Project Structure

```text
TCDV/
├── backend/
│   ├── app.py                  # Flask REST API backend
│   ├── quality_filter.py       # Image blur & exposure assessment
│   └── requirements.txt        # Backend dependencies
├── frontend/
│   ├── index.html              # Responsive web dashboard
│   └── app.js                  # Frontend client logic & API bindings
├── runs/                       # Trained YOLOv8 model weights & plots
│   └── detect/runs/tower_detect/weights/
│       ├── best.pt             # Optimal trained weights (6.2 MB)
│       └── last.pt             # Checkpoint weights
├── dataset/
│   ├── labels/                 # YOLO format ground truth bounding boxes
│   └── data.yaml               # YOLO dataset configuration
├── prepare_dataset.py          # Multimodal dataset preparation & auto-labeling
├── check_labels.py             # Visual bounding box verification utility
├── quality_filter.py           # Standalone quality assessment script
├── train.py                    # YOLOv8 model training script
├── evaluate.py                 # Validation and metric generation script
├── evaluation_metrics.json     # Saved evaluation metrics
└── README.md
```

---

## ⚡ Quickstart

### 1. Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/sujay2520/tower-component-detection.git
cd tower-component-detection
pip install -r backend/requirements.txt
```

### 2. Run the Backend API
```bash
cd backend
python app.py
```
The Flask backend starts at `http://localhost:5000`.

### 3. Run the Frontend Dashboard
In a separate terminal:
```bash
cd frontend
python -m http.server 8080
```
Open [http://localhost:8080](http://localhost:8080) in your web browser.

---

## 🔌 API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/health` | `GET` | Service status and verified model weights path |
| `/upload` | `POST` | Upload target image for inference |
| `/execute` | `POST` | Execute quality filter and run YOLOv8 object detection |
| `/outputs/<filename>` | `GET` | Retrieve detection output image with bounding box overlays |
| `/metrics` | `GET` | Fetch evaluation metrics JSON |

---

## 🛠️ Tech Stack

- **Model**: Ultralytics YOLOv8
- **Computer Vision**: OpenCV, PIL, NumPy
- **Backend**: Python, Flask, Flask-CORS
- **Frontend**: Vanilla HTML5, CSS3, Modern ES6 JavaScript
