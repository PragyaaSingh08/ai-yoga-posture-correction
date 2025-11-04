# =========================================
# build_dataset.py
# Build full training dataset from yoga images/videos
# Extracts YOLO keypoints + computed joint angles
# =========================================

import os
import cv2
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

# -----------------------------
# 1️⃣ Configuration
# -----------------------------
DATASET_DIR = "dataset"          # Folder with yoga videos/images (organize by posture/label if possible)
OUTPUT_DIR = "outputs"
MODEL_PATH = "yolov8n-pose.pt"   # Lightweight YOLOv8 pose model
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load YOLOv8 pose model
model = YOLO(MODEL_PATH)

# -----------------------------
# 2️⃣ Helper: Compute joint angle
# -----------------------------
def compute_angle(A, B, C):
    BA = A - B
    BC = C - B
    cos_angle = np.dot(BA, BC) / (np.linalg.norm(BA) * np.linalg.norm(BC) + 1e-8)
    return np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))

# -----------------------------
# 3️⃣ Collect dataset files
# -----------------------------
file_types = ("*.mp4", "*.avi", "*.mov", "*.jpg", "*.jpeg", "*.png")
dataset_files = []
for ext in file_types:
    dataset_files.extend(glob.glob(os.path.join(DATASET_DIR, "**", ext), recursive=True))

if not dataset_files:
    raise FileNotFoundError("❌ No images or videos found in 'dataset/' folder.")

print(f"📂 Found {len(dataset_files)} files for dataset building")

# -----------------------------
# 4️⃣ Process each file
# -----------------------------
all_rows = []

for file_path in tqdm(dataset_files, desc="Extracting keypoints and angles"):
    label = os.path.basename(os.path.dirname(file_path))  # Folder name = posture label

    # Load frames depending on type
    if file_path.lower().endswith((".jpg", ".jpeg", ".png")):
        frames = [cv2.imread(file_path)]
    else:
        cap = cv2.VideoCapture(file_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

    # Process frames with YOLO
    for frame in frames[::5]:  # every 5th frame for efficiency
        results = model(frame, verbose=False)
        if not results or len(results[0].keypoints.xy) == 0:
            continue

        keypoints = results[0].keypoints.xy[0].cpu().numpy()  # (17, 2)
        if keypoints.shape[0] < 17:
            continue

        # Compute angles
        try:
            left_knee = compute_angle(keypoints[11], keypoints[13], keypoints[15])
            right_knee = compute_angle(keypoints[12], keypoints[14], keypoints[16])
            left_elbow = compute_angle(keypoints[5], keypoints[7], keypoints[9])
            right_elbow = compute_angle(keypoints[6], keypoints[8], keypoints[10])
            left_shoulder = compute_angle(keypoints[7], keypoints[5], keypoints[11])
            right_shoulder = compute_angle(keypoints[8], keypoints[6], keypoints[12])
        except Exception:
            continue

        all_rows.append({
            "file": os.path.basename(file_path),
            "label": label,
            "left_knee": left_knee,
            "right_knee": right_knee,
            "left_elbow": left_elbow,
            "right_elbow": right_elbow,
            "left_shoulder": left_shoulder,
            "right_shoulder": right_shoulder
        })

# -----------------------------
# 5️⃣ Save dataset
# -----------------------------
df = pd.DataFrame(all_rows)
output_csv = os.path.join(OUTPUT_DIR, "full_dataset_angles.csv")
df.to_csv(output_csv, index=False)
print(f"\n✅ Dataset built successfully!")
print(f"📄 Saved at: {output_csv}")
print(f"🧩 Total samples: {len(df)}")
print(f"📊 Labels found: {df['label'].unique().tolist()}")
