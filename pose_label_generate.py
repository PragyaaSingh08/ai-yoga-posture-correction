# =========================================
# pose_label_generate.py (light version)
# =========================================

import os
import cv2
from ultralytics import YOLO
import pandas as pd
from tqdm import tqdm

FRAMES_DIR = "frames_dataset"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

model = YOLO("yolov8n-pose.pt")  # Lightweight model

data = []
max_frames = 15  # ✅ Limit frames per pose

for folder in tqdm(sorted(os.listdir(FRAMES_DIR))):
    folder_path = os.path.join(FRAMES_DIR, folder)
    if not os.path.isdir(folder_path):
        continue

    label = folder.split("_")[-1]  # e.g. Bhujangasana
    frame_files = sorted(os.listdir(folder_path))[:max_frames]  # limit frames

    for frame_name in frame_files:
        frame_path = os.path.join(folder_path, frame_name)
        try:
            results = model(frame_path, verbose=False)[0]
            keypoints = results.keypoints.xy.cpu().numpy()[0]  # shape (17,2)
            frame_data = {"pose": label, "frame": frame_name}
            for i, (x, y) in enumerate(keypoints):
                frame_data[f"x{i}"] = x
                frame_data[f"y{i}"] = y
            data.append(frame_data)
        except Exception as e:
            print(f"⚠️ Error with {frame_name}: {e}")

df = pd.DataFrame(data)
csv_path = os.path.join(OUTPUT_DIR, "pose_keypoints.csv")
df.to_csv(csv_path, index=False)
print(f"✅ Keypoints saved at {csv_path}")
