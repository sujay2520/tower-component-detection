# Tower_Detection_Training.ipynb - Run in Google Colab
# =============================================================
# Upload this as a .py file to Colab, or copy cells into a notebook
#
# STEP 1: Mount Google Drive
# STEP 2: Upload dataset_merged/ folder to Drive
# STEP 3: Run training
# =============================================================

# ---- Cell 1: Setup ----
# !pip install ultralytics>=8.2.0

# ---- Cell 2: Mount Drive ----
# from google.colab import drive
# drive.mount('/content/drive')

# ---- Cell 3: Check GPU ----
# import torch
# print(f"CUDA: {torch.cuda.is_available()}")
# print(f"GPU: {torch.cuda.get_device_name(0)}")

# ---- Cell 4: Train ----
# from ultralytics import YOLO
# model = YOLO("yolov8n.pt")
# model.train(
#     data="/content/drive/MyDrive/tower_detection/dataset_merged/data.yaml",
#     epochs=100, imgsz=640, batch=16, patience=20,
#     project="/content/runs", name="tower_final",
#     device=0, augment=True, mosaic=1.0, mixup=0.15,
#     flipud=0.5, fliplr=0.5, degrees=20, scale=0.5,
#     hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
#     optimizer="AdamW", lr0=0.01, lrf=0.01,
#     warmup_epochs=5, plots=True, save=True, val=True,
# )

# ---- Cell 5: Evaluate ----
# model = YOLO("/content/runs/tower_final/weights/best.pt")
# metrics = model.val(data="/content/drive/MyDrive/tower_detection/dataset_merged/data.yaml")
# print(f"mAP50: {metrics.box.map50:.4f}")
# print(f"Precision: {metrics.box.mp:.4f}")
# print(f"Recall: {metrics.box.mr:.4f}")

# ---- Cell 6: Copy weights back to Drive ----
# import shutil
# shutil.copy("/content/runs/tower_final/weights/best.pt",
#             "/content/drive/MyDrive/tower_detection/best_tower_model.pt")
# print("Model saved to Google Drive!")

print("This file contains the Colab notebook instructions.")
print("Copy the cells above into a Google Colab notebook to train on free GPU.")
print("Or run train_colab.py directly for local training.")
