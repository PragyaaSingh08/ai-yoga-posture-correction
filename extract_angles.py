# =========================================
# extract_angles.py (compatible with pose_label_generate.py light version)
# =========================================

import os
import glob
import pandas as pd
import numpy as np

# Folder where pose_keypoints.csv is saved
OUTPUT_DIR = "outputs"

# -----------------------------
# 1️⃣ Find latest CSV in outputs
# -----------------------------
csv_files = glob.glob(os.path.join(OUTPUT_DIR, "*_keypoints.csv"))
if not csv_files:
    raise FileNotFoundError("No keypoints CSV found in outputs/")

latest_csv = max(csv_files, key=os.path.getmtime)
print(f"✅ Using latest CSV: {latest_csv}")

# -----------------------------
# 2️⃣ Load CSV safely
# -----------------------------
df = pd.read_csv(latest_csv)
columns = df.columns.tolist()
print(f"📄 Columns in CSV: {columns[:10]} ...")  # preview first few columns

# -----------------------------
# 3️⃣ Define helper function
# -----------------------------
def compute_angle(A, B, C):
    """Compute angle ABC in degrees given 3 points (2D)."""
    BA = A - B
    BC = C - B
    cos_angle = np.dot(BA, BC) / (np.linalg.norm(BA) * np.linalg.norm(BC) + 1e-8)
    return np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))

# -----------------------------
# 4️⃣ Compute joint angles
# -----------------------------
x_cols = [c for c in df.columns if c.startswith('x')]
y_cols = [c for c in df.columns if c.startswith('y')]

angles_list = []

for idx, row in df.iterrows():
    xs = row[x_cols].to_numpy()
    ys = row[y_cols].to_numpy()

    if len(xs) < 17 or len(ys) < 17:
        continue  # skip incomplete frames

    points = np.stack([xs, ys], axis=1)

    try:
        left_knee = compute_angle(points[11], points[13], points[15])
        right_knee = compute_angle(points[12], points[14], points[16])
        left_elbow = compute_angle(points[5], points[7], points[9])
        right_elbow = compute_angle(points[6], points[8], points[10])
        left_shoulder = compute_angle(points[7], points[5], points[11])
        right_shoulder = compute_angle(points[8], points[6], points[12])
    except Exception as e:
        print(f"⚠️ Skipping frame {idx}: {e}")
        continue

    angles_list.append({
        "frame": row["frame"],
        "pose": row["pose"],
        "left_knee": left_knee,
        "right_knee": right_knee,
        "left_elbow": left_elbow,
        "right_elbow": right_elbow,
        "left_shoulder": left_shoulder,
        "right_shoulder": right_shoulder
    })

angles_df = pd.DataFrame(angles_list)

# -----------------------------
# 5️⃣ Save output
# -----------------------------
angles_csv_path = os.path.join(OUTPUT_DIR, f"angles_{os.path.basename(latest_csv)}")
angles_df.to_csv(angles_csv_path, index=False)
print(f"✅ Angles CSV saved at: {angles_csv_path}")
print(f"📊 Total frames processed: {len(angles_df)}")
