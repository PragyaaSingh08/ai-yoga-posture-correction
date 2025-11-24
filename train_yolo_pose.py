# ================================================================
# Yoga Pose Classification using YOLOv12n
# ================================================================

import os
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from ultralytics import YOLO
from ultralytics.utils import LOGGER
import numpy as np

# ================================================================
# STEP 1: Configuration
# ================================================================
MODEL_PATH = r"C:\Users\DR. Sindhu Sagar\Desktop\(79) WhatsApp_files\yoga posture\yolov12n-cls.pt"
DATASET_PATH = r"C:\Users\DR. Sindhu Sagar\Desktop\(79) WhatsApp_files\yoga posture\dataset"
RESULTS_DIR = os.path.join(DATASET_PATH, "results")

# Auto-create results directory
os.makedirs(RESULTS_DIR, exist_ok=True)

# Select device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Using device: {device}")

# ================================================================
# STEP 2: Load YOLOv12 Classification Model
# ================================================================
print("\n🔍 Loading YOLOv12n model...")
model = YOLO(MODEL_PATH)

# ================================================================
# STEP 3: Train the Modelye
# ================================================================
print("\n🏋️ Training started...")
results = model.train(
    data=DATASET_PATH,
    epochs=60,
    imgsz=224,
    batch=16,
    device=device,
    verbose=True
)

# ================================================================
# STEP 4: Validate the Model
# ================================================================
print("\n📊 Evaluating model on validation set...")
metrics = model.val(data=DATASET_PATH, device=device)

# ================================================================
# STEP 5: Display Evaluation Metrics
# ================================================================
print("\n================ Evaluation Metrics ================")
for key, value in metrics.results_dict.items():
    print(f"{key:25s}: {value:.4f}")

# ================================================================
# STEP 6: Confusion Matrix Plot (Safe)
# ================================================================
try:
    cm = metrics.confusion_matrix  # Fetch confusion matrix
    if isinstance(cm, np.ndarray) and cm.ndim == 2:
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=model.names.values(),
            yticklabels=model.names.values()
        )
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")
        plt.tight_layout()

        # Save confusion matrix
        os.makedirs(RESULTS_DIR, exist_ok=True)
        plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"))
        plt.close()
        print(f"\n✅ Confusion matrix saved to {RESULTS_DIR}\\confusion_matrix.png")
    else:
        LOGGER.warning("⚠️ Confusion matrix is empty or invalid, skipping heatmap.")
except Exception as e:
    LOGGER.warning(f"⚠️ Could not plot confusion matrix: {e}")

# ================================================================
# STEP 7: Display Summary
# ================================================================
print("\n✅ Training & Evaluation Complete!")
print(f"📁 All results saved in: {model.trainer.save_dir}")
