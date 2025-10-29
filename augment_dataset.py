# =========================================
# augment_dataset.py
# Augment yoga dataset images for better model training
# Author: Pragya Singh
# =========================================

import os
import cv2
import random
from tqdm import tqdm
import numpy as np

# -----------------------------
# Paths
# -----------------------------
DATASET_DIR = "dataset"               # original dataset
AUGMENTED_DIR = "augmented_dataset"   # output folder

# Create folders
os.makedirs(AUGMENTED_DIR, exist_ok=True)
for label in ["correct", "incorrect"]:
    os.makedirs(os.path.join(AUGMENTED_DIR, label), exist_ok=True)

# -----------------------------
# Augmentation functions
# -----------------------------
def rotate_image(image):
    angle = random.choice([-15, -10, -5, 5, 10, 15])
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
    return cv2.warpAffine(image, M, (w, h))

def flip_image(image):
    return cv2.flip(image, 1)  # horizontal flip

def change_brightness(image):
    value = random.randint(-40, 40)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] + value, 0, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def add_noise(image):
    noise = np.random.normal(0, 25, image.shape).astype(np.uint8)
    return cv2.add(image, noise)

def blur_image(image):
    return cv2.GaussianBlur(image, (5, 5), 0)

AUG_FUNCTIONS = [rotate_image, flip_image, change_brightness, add_noise, blur_image]

# -----------------------------
# Augment all images
# -----------------------------
count = 0
for label in ["correct", "incorrect"]:
    folder = os.path.join(DATASET_DIR, label)
    output_folder = os.path.join(AUGMENTED_DIR, label)

    if not os.path.exists(folder):
        print(f"⚠️ Skipping missing folder: {folder}")
        continue

    images = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"🧘 Augmenting {len(images)} images in '{label}' class...")

    for img_name in tqdm(images, desc=f"Augmenting {label}"):
        img_path = os.path.join(folder, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        # Save original copy also (optional)
        base_name = os.path.splitext(img_name)[0]
        cv2.imwrite(os.path.join(output_folder, f"{base_name}_orig.jpg"), img)

        # Apply all augmentations
        for func in AUG_FUNCTIONS:
            aug_img = func(img)
            aug_name = f"{base_name}_{func.__name__}.jpg"
            cv2.imwrite(os.path.join(output_folder, aug_name), aug_img)
            count += 1

print(f"\n✅ Data augmentation complete!")
print(f"✅ Total augmented images created: {count}")
print(f"📁 Saved in: {AUGMENTED_DIR}/")
