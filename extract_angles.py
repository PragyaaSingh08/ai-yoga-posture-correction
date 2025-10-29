# =========================================
# extract_angles.py
# Robust angle extraction from YOLOv8 CSV
# Supports both raw keypoints and pre-computed angles
# =========================================

import os
import glob
import pandas as pd
import numpy as np

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
df = pd.read_csv(latest_csv, index_col=0, engine='python')
columns = df.columns.tolist()
print(f"Columns in CSV: {columns}")

# -----------------------------
# 3️⃣ Detect type of CSV
# -----------------------------
angle_cols = ['left_knee', 'right_knee', 'left_elbow', 'right_elbow', 'left_shoulder', 'right_shoulder']
x_cols = [c for c in columns if c.startswith('x')]
y_cols = [c for c in columns if c.startswith('y')]

if set(angle_cols).issubset(columns):
    print("✅ CSV already contains angles. No computation needed.")
    angles_df = df[angle_cols + ['frame']] if 'frame' in columns else df[angle_cols]
else:
    if not x_cols or not y_cols:
        raise ValueError("No x/y columns detected in CSV. Cannot compute angles.")
    print(f"Detected {len(x_cols)} keypoints per person. Computing angles...")

    num_keypoints = len(x_cols)

    # Angle calculation function
    def compute_angle(A, B, C):
        BA = A - B
        BC = C - B
        cos_angle = np.dot(BA, BC) / (np.linalg.norm(BA)*np.linalg.norm(BC) + 1e-8)
        return np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))

    angles_list = []
    for idx, row in df.iterrows():
        xs = row[x_cols].to_numpy()
        ys = row[y_cols].to_numpy()

        if len(xs) != num_keypoints or len(ys) != num_keypoints:
            continue  # skip incomplete rows

        points = np.stack([xs, ys], axis=1)

        try:
            left_knee = compute_angle(points[11], points[13], points[15])
            right_knee = compute_angle(points[12], points[14], points[16])
            left_elbow = compute_angle(points[5], points[7], points[9])
            right_elbow = compute_angle(points[6], points[8], points[10])
            left_shoulder = compute_angle(points[7], points[5], points[11])
            right_shoulder = compute_angle(points[8], points[6], points[12])
        except IndexError:
            continue

        angles_list.append({
            "frame": row["frame"] if "frame" in columns else idx,
            "left_knee": left_knee,
            "right_knee": right_knee,
            "left_elbow": left_elbow,
            "right_elbow": right_elbow,
            "left_shoulder": left_shoulder,
            "right_shoulder": right_shoulder
        })

    angles_df = pd.DataFrame(angles_list)

# -----------------------------
# 4️⃣ Save angles CSV
# -----------------------------
angles_csv_path = os.path.join(OUTPUT_DIR, f"angles_{os.path.basename(latest_csv)}")
angles_df.to_csv(angles_csv_path, index=False)
print(f"✅ Angles CSV saved at {angles_csv_path}")
