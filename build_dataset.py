# =========================================
# build_dataset.py
# Create full dataset of angles + labels from all augmented yoga images/videos
# =========================================

import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from tqdm import tqdm

# -----------------------------
# Paths (updated)
# -----------------------------
DATASET_DIR = "augmented_dataset"   # 👈 use augmented dataset instead of raw
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load YOLOv8 pose model (lightweight version)
model = YOLO("yolov8n-pose.pt")

# -----------------------------
# Angle computation
# -----------------------------
def compute_angle(A, B, C):
    BA = A - B
    BC = C - B
    cos_angle = np.dot(BA, BC) / (np.linalg.norm(BA)*np.linalg.norm(BC) + 1e-8)
    return np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))

def process_frame(keypoints):
    """Compute 6 angles from a set of 17 keypoints"""
    points = np.array(keypoints).reshape(-1, 2)
    try:
        left_knee = compute_angle(points[11], points[13], points[15])
        right_knee = compute_angle(points[12], points[14], points[16])
        left_elbow = compute_angle(points[5], points[7], points[9])
        right_elbow = compute_angle(points[6], points[8], points[10])
        left_shoulder = compute_angle(points[7], points[5], points[11])
        right_shoulder = compute_angle(points[8], points[6], points[12])
    except IndexError:
        return None
    return [left_knee, right_knee, left_elbow, right_elbow, left_shoulder, right_shoulder]

def process_file(filepath, label):
    """Process a single image or video file"""
    ext = filepath.lower().split('.')[-1]
    data_rows = []

    if ext in ['jpg', 'jpeg', 'png']:
        img = cv2.imread(filepath)
        if img is None:
            return []
        results = model(img, verbose=False)
        for r in results:
            if r.keypoints is not None and len(r.keypoints.xy) > 0:
                keypoints = r.keypoints.xy[0].cpu().numpy().flatten()
                angles = process_frame(keypoints)
                if angles:
                    data_rows.append(angles + [label])

    elif ext in ['mp4', 'mov', 'avi']:
        cap = cv2.VideoCapture(filepath)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame, verbose=False)
            for r in results:
                if r.keypoints is not None and len(r.keypoints.xy) > 0:
                    keypoints = r.keypoints.xy[0].cpu().numpy().flatten()
                    angles = process_frame(keypoints)
                    if angles:
                        data_rows.append(angles + [label])
        cap.release()

    return data_rows


# -----------------------------
# Process all files
# -----------------------------
all_files = []
for label in ["correct", "incorrect"]:
    folder = os.path.join(DATASET_DIR, label)
    if os.path.exists(folder):
        for f in os.listdir(folder):
            if f.lower().endswith(('.mp4', '.mov', '.avi', '.jpg', '.jpeg', '.png')):
                all_files.append((os.path.join(folder, f), label))

print(f"🧘 Processing {len(all_files)} files from '{DATASET_DIR}' ...")

all_data = []
for filepath, label in tqdm(all_files, desc="Extracting Angles"):
    data_rows = process_file(filepath, label)
    all_data.extend(data_rows)

# -----------------------------
# Save dataset
# -----------------------------
columns = ['left_knee', 'right_knee', 'left_elbow', 'right_elbow', 'left_shoulder', 'right_shoulder', 'label']
df = pd.DataFrame(all_data, columns=columns)

dataset_csv = os.path.join(OUTPUT_DIR, "full_dataset_angles.csv")
df.to_csv(dataset_csv, index=False)

print(f"\n✅ Yoga angle dataset built successfully!")
print(f"✅ Total samples: {len(df)}")
print(f"📁 Saved at: {dataset_csv}")
