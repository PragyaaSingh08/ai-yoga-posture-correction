import pandas as pd
import numpy as np
import os

# === 1️⃣ Load original data ===
INPUT_FILE = "outputs/angles_pose_keypoints.csv"
OUTPUT_FILE = "outputs/augmented_angles.csv"

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"❌ File not found: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)
print(f"✅ Loaded: {INPUT_FILE}")
print(f"Columns: {df.columns.tolist()}")

# === 2️⃣ Separate features and labels ===
# Drop non-numeric columns (like 'pose', 'frame')
features = df.drop(["pose", "frame"], axis=1)
labels = df["pose"]

# === 3️⃣ Define augmentation functions ===
def add_noise(X, noise_level=0.05):
    noise = np.random.normal(0, noise_level, X.shape)
    return X + noise

def scale(X, scale_range=(0.9, 1.1)):
    factor = np.random.uniform(scale_range[0], scale_range[1])
    return X * factor

def rotate(X, angle_range=(-5, 5)):
    # simulate slight angular deviation (small rotations)
    angle = np.random.uniform(angle_range[0], angle_range[1])
    radians = np.deg2rad(angle)
    return X * np.cos(radians)

# === 4️⃣ Apply augmentations ===
augmented_data = []

for i in range(len(features)):
    original = features.iloc[i].values

    # random choice of augmentation type
    aug_type = np.random.choice(["noise", "scale", "rotate"])

    if aug_type == "noise":
        new_sample = add_noise(original)
    elif aug_type == "scale":
        new_sample = scale(original)
    else:
        new_sample = rotate(original)

    augmented_data.append(new_sample)

# Convert to DataFrame
aug_df = pd.DataFrame(augmented_data, columns=features.columns)
aug_df["pose"] = labels.values  # keep same labels

# Combine with original data
final_df = pd.concat([df, aug_df], ignore_index=True)

# === 5️⃣ Save the augmented dataset ===
os.makedirs("outputs", exist_ok=True)
final_df.to_csv(OUTPUT_FILE, index=False)

print(f"✅ Data augmentation complete!")
print(f"Original samples: {len(df)} → Augmented: {len(final_df)}")
print(f"📁 Saved: {OUTPUT_FILE}")
