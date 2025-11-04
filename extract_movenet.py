# ============================================================
# MoveNet Keypoint Extraction (Debug-Safe Version)
# ============================================================

import os
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd

# ✅ Load MoveNet model from TensorFlow Hub
movenet = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")

# ✅ Path to your dataset
DATASET_DIR = "dataset"   # Change if needed
OUTPUT_PATH = "outputs/movenet_keypoints.csv"
os.makedirs("outputs", exist_ok=True)

all_data, poses = [], []

# ✅ Check if dataset folder exists
if not os.path.exists(DATASET_DIR):
    raise FileNotFoundError(f"❌ Dataset folder not found: {DATASET_DIR}")

# ✅ Iterate through folders
for pose_name in os.listdir(DATASET_DIR):
    pose_folder = os.path.join(DATASET_DIR, pose_name)
    if not os.path.isdir(pose_folder):
        continue

    print(f"\n🔍 Extracting keypoints for pose: {pose_name}")
    image_files = [f for f in os.listdir(pose_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if not image_files:
        print(f"⚠️ No images found in {pose_folder}")
        continue

    for img_file in image_files:
        img_path = os.path.join(pose_folder, img_file)
        img = cv2.imread(img_path)

        if img is None:
            print(f"⚠️ Skipping unreadable image: {img_path}")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = tf.image.resize_with_pad(tf.expand_dims(img_rgb, axis=0), 192, 192)
        input_img = tf.cast(img_tensor, dtype=tf.int32)

        # Run inference
        outputs = movenet.signatures['serving_default'](input_img)
        keypoints = outputs['output_0'].numpy()[0, 0, :, :2].flatten()

        all_data.append(keypoints)
        poses.append(pose_name)

# ✅ Final check before saving
if len(all_data) == 0:
    raise RuntimeError("❌ No keypoints were extracted. Check if dataset images exist and are readable.")

# ✅ Convert to DataFrame
columns = [f"kp_{i}" for i in range(len(all_data[0]))]
df = pd.DataFrame(all_data, columns=columns)
df["pose"] = poses

df.to_csv(OUTPUT_PATH, index=False)
print(f"\n✅ Keypoints extracted for {len(df)} images.")
print(f"📁 Saved to: {OUTPUT_PATH}")
