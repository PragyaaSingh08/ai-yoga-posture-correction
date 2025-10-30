import albumentations as A
import cv2
import os
from tqdm import tqdm

input_folder = "dataset"
output_folder = "augmented"
num_augmented_per_image = 5

transform = A.Compose([
    A.RandomRotate90(p=0.5),
    A.Flip(p=0.5),
    A.Transpose(p=0.3),
    A.RandomBrightnessContrast(p=0.5),
    A.HueSaturationValue(p=0.3),
    A.GaussNoise(p=0.3),
    A.MotionBlur(p=0.2),
    A.RandomResizedCrop(height=224, width=224, scale=(0.8, 1.0), p=0.4),
])

for root, dirs, files in os.walk(input_folder):
    for img_name in tqdm(files, desc=f"Augmenting in {os.path.basename(root)}"):
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        img_path = os.path.join(root, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        # Create matching output subfolder
        relative_path = os.path.relpath(root, input_folder)
        output_subfolder = os.path.join(output_folder, relative_path)
        os.makedirs(output_subfolder, exist_ok=True)

        for i in range(num_augmented_per_image):
            augmented = transform(image=img)
            aug_img = augmented['image']
            base_name = os.path.splitext(img_name)[0]
            aug_name = f"{base_name}_aug_{i+1}.jpg"
            cv2.imwrite(os.path.join(output_subfolder, aug_name), aug_img)

print("\n✅ All augmented images saved inside 'augmented/' with the same folder structure.")
