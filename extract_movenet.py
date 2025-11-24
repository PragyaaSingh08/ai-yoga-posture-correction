# ==============================================================
# extract_movenet_auto_fixed.py  ✅ Full Robust Version
# ==============================================================

import os
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm
import glob
import random
import shutil

# --------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------
DATASET_DIR = "dataset"   # base dataset (with train/val subfolders)
OUTPUT_DIR = "keypoints"  # where keypoints will be saved
MIN_DETECTION_CONFIDENCE = 0.4
VAL_COPY_COUNT = 10       # if val folder is empty, copy this many samples from train

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------------------
# 1️⃣ NORMALIZE FOLDER NAMES (lowercase, underscores)
# --------------------------------------------------------------
def normalize_folders(base_path):
    for subset in ["train", "val"]:
        subset_path = os.path.join(base_path, subset)
        if not os.path.exists(subset_path):
            continue
        for folder in os.listdir(subset_path):
            old_path = os.path.join(subset_path, folder)
            if not os.path.isdir(old_path):
                continue
            new_name = folder.strip().lower().replace(" ", "_")
            new_path = os.path.join(subset_path, new_name)
            if old_path != new_path:
                os.rename(old_path, new_path)
                print(f"🔤 Renamed '{folder}' → '{new_name}'")

normalize_folders(DATASET_DIR)

# --------------------------------------------------------------
# 2️⃣ FILL EMPTY VAL FOLDERS (copy few images from train)
# --------------------------------------------------------------
def ensure_val_images(base_path):
    train_path = os.path.join(base_path, "train")
    val_path = os.path.join(base_path, "val")
    if not (os.path.exists(train_path) and os.path.exists(val_path)):
        return

    for cls in os.listdir(train_path):
        train_cls = os.path.join(train_path, cls)
        val_cls = os.path.join(val_path, cls)
        os.makedirs(val_cls, exist_ok=True)

        # Check if val has any images
        val_imgs = glob.glob(os.path.join(val_cls, "*.*"))
        if len(val_imgs) == 0:
            train_imgs = glob.glob(os.path.join(train_cls, "*.*"))
            if len(train_imgs) == 0:
                continue

            sample_imgs = random.sample(train_imgs, min(VAL_COPY_COUNT, len(train_imgs)))
            for img_path in sample_imgs:
                shutil.copy(img_path, val_cls)
            print(f"📸 Added {len(sample_imgs)} images to val/{cls}")

ensure_val_images(DATASET_DIR)

# --------------------------------------------------------------
# 3️⃣ MEDIAPIPE POSE SETUP
# --------------------------------------------------------------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=MIN_DETECTION_CONFIDENCE)
mp_drawing = mp.solutions.drawing_utils

# --------------------------------------------------------------
# Extract all 33 landmarks (x, y, z)
# --------------------------------------------------------------
def extract_keypoints(landmarks):
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()

# --------------------------------------------------------------
# 4️⃣ MAIN EXTRACTION LOOP
# --------------------------------------------------------------
sets = ["train", "val"]

for subset in sets:
    subset_path = os.path.join(DATASET_DIR, subset)
    if not os.path.exists(subset_path):
        print(f"⚠️ Skipping {subset} (folder not found).")
        continue

    all_keypoints = []
    all_labels = []

    classes = sorted(os.listdir(subset_path))
    print(f"\n📂 Processing {subset.upper()} set ({len(classes)} classes)...")

    for class_name in classes:
        class_path = os.path.join(subset_path, class_name)
        if not os.path.isdir(class_path):
            continue

        images = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
            images.extend(glob.glob(os.path.join(class_path, ext)))

        if len(images) == 0:
            print(f"⚠️ No images found in {class_path}")
            continue

        for img_path in tqdm(images, desc=f"🧘 {class_name}", ncols=80):
            img = cv2.imread(img_path)
            if img is None:
                print(f"❌ Could not read: {img_path}")
                continue

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            if not results.pose_landmarks:
                continue

            keypoints = extract_keypoints(results.pose_landmarks.landmark)
            all_keypoints.append(keypoints)
            all_labels.append(class_name)

    # ----------------------------------------------------------
    # Save Numpy arrays
    # ----------------------------------------------------------
    if len(all_keypoints) == 0:
        print(f"⚠️ No keypoints extracted for {subset}.")
        continue

    np.save(os.path.join(OUTPUT_DIR, f"{subset}_keypoints.npy"), np.array(all_keypoints, dtype=np.float32))
    np.save(os.path.join(OUTPUT_DIR, f"{subset}_labels.npy"), np.array(all_labels))

    print(f"✅ Saved {len(all_keypoints)} samples for {subset} set → {OUTPUT_DIR}/")

pose.close()
print("\n🎉 Keypoint extraction complete for all sets!")
