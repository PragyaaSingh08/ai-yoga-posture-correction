import os
import shutil
import random

# === CONFIGURATION ===
DATASET_DIR = r"C:\Users\DR. Sindhu Sagar\Desktop\(79) WhatsApp_files\yoga posture\dataset"
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "val")
TEST_DIR = os.path.join(DATASET_DIR, "test")

SPLIT_RATIO = 0.8  # 80% train, 20% val

# === SETUP FOLDERS ===
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

# === LOOP OVER POSE FOLDERS ===
for pose_name in os.listdir(DATASET_DIR):
    pose_path = os.path.join(DATASET_DIR, pose_name)

    # Skip known folders
    if pose_name.lower() in ["train", "val", "test"]:
        continue
    if not os.path.isdir(pose_path):
        continue

    # List all images
    images = [f for f in os.listdir(pose_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if len(images) == 0:
        print(f"⚠️ No images found in {pose_name}, skipping.")
        continue

    random.shuffle(images)
    split_index = int(len(images) * SPLIT_RATIO)
    train_imgs = images[:split_index]
    val_imgs = images[split_index:]

    # Create corresponding class folders
    os.makedirs(os.path.join(TRAIN_DIR, pose_name), exist_ok=True)
    os.makedirs(os.path.join(VAL_DIR, pose_name), exist_ok=True)

    # Move images
    for img in train_imgs:
        shutil.move(os.path.join(pose_path, img), os.path.join(TRAIN_DIR, pose_name, img))
    for img in val_imgs:
        shutil.move(os.path.join(pose_path, img), os.path.join(VAL_DIR, pose_name, img))

    # Delete empty source folder
    if not os.listdir(pose_path):
        os.rmdir(pose_path)

    print(f"✅ Split '{pose_name}': {len(train_imgs)} → train, {len(val_imgs)} → val")

print("\n🎯 Dataset ready for YOLO classification!")
print(f"Train path: {TRAIN_DIR}")
print(f"Val path:   {VAL_DIR}")
print(f"Test path:  {TEST_DIR} (unchanged)")
