# ============================================================
# generate_pose_angles.py ✅ Converts keypoints → joint angles
# ============================================================

import os
import pandas as pd
import numpy as np
import math

# Input and output paths
INPUT_CSV = "outputs/pose_keypoints.csv"
OUTPUT_CSV = "outputs/augmented_angles.csv"

if not os.path.exists(INPUT_CSV):
    raise FileNotFoundError(f"❌ Keypoints file not found: {INPUT_CSV}")

print(f"📂 Loading keypoints from: {INPUT_CSV}")
df = pd.read_csv(INPUT_CSV)

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
def calculate_angle(a, b, c):
    """Calculate the angle (in degrees) between points a, b, c."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def get_point(row, i):
    """Return (x, y) for a specific keypoint index."""
    return (row[f"x{i}"], row[f"y{i}"])

# ------------------------------------------------------------
# Keypoint index reference (for MoveNet/YOLOv8-Pose)
# ------------------------------------------------------------
# 0: nose
# 1: left_eye
# 2: right_eye
# 3: left_ear
# 4: right_ear
# 5: left_shoulder
# 6: right_shoulder
# 7: left_elbow
# 8: right_elbow
# 9: left_wrist
# 10: right_wrist
# 11: left_hip
# 12: right_hip
# 13: left_knee
# 14: right_knee
# 15: left_ankle
# 16: right_ankle

# ------------------------------------------------------------
# Compute angles
# ------------------------------------------------------------
angle_data = []

for _, row in df.iterrows():
    try:
        angles = {}

        # Shoulder-Elbow-Wrist
        angles["left_elbow_angle"] = calculate_angle(get_point(row, 5), get_point(row, 7), get_point(row, 9))
        angles["right_elbow_angle"] = calculate_angle(get_point(row, 6), get_point(row, 8), get_point(row, 10))

        # Hip-Knee-Ankle
        angles["left_knee_angle"] = calculate_angle(get_point(row, 11), get_point(row, 13), get_point(row, 15))
        angles["right_knee_angle"] = calculate_angle(get_point(row, 12), get_point(row, 14), get_point(row, 16))

        # Shoulder-Hip-Knee
        angles["left_hip_angle"] = calculate_angle(get_point(row, 5), get_point(row, 11), get_point(row, 13))
        angles["right_hip_angle"] = calculate_angle(get_point(row, 6), get_point(row, 12), get_point(row, 14))

        # Optional: torso and arm-leg combined angles
        angles["shoulder_angle"] = calculate_angle(get_point(row, 5), get_point(row, 6), get_point(row, 12))
        angles["hip_angle"] = calculate_angle(get_point(row, 11), get_point(row, 12), get_point(row, 6))

        # Pose and frame metadata
        angles["pose"] = row["pose"]
        angles["frame"] = row["frame"]

        angle_data.append(angles)

    except Exception as e:
        print(f"⚠️ Skipped one row due to error: {e}")
        continue

# ------------------------------------------------------------
# Convert to DataFrame
# ------------------------------------------------------------
angle_df = pd.DataFrame(angle_data)
print(f"✅ Computed {len(angle_df)} valid rows of angles")

# ------------------------------------------------------------
# Data augmentation (optional — adds small random noise)
# ------------------------------------------------------------
augmented_df = angle_df.copy()
noise_level = 2.0  # degrees
numeric_cols = [c for c in angle_df.columns if c not in ["pose", "frame"]]

for _ in range(3):  # create 3x more data
    noisy = angle_df.copy()
    noisy[numeric_cols] += np.random.normal(0, noise_level, size=noisy[numeric_cols].shape)
    augmented_df = pd.concat([augmented_df, noisy], ignore_index=True)

print(f"📈 After augmentation: {len(augmented_df)} samples")

# ------------------------------------------------------------
# Save output CSV
# ------------------------------------------------------------
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
augmented_df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Angles + Augmented data saved at: {OUTPUT_CSV}")
