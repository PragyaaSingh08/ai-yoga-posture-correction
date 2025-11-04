# =====================================================
# generate_dataset_yaml.py
# =====================================================
# Automatically create a YOLO classification YAML file
# based on your dataset folder structure.
# =====================================================

import os
import yaml

# 🔹 Path to your dataset root folder
dataset_root = r"C:\Users\DR. Sindhu Sagar\Desktop\(79) WhatsApp_files\yoga posture\dataset"

# 🔹 Paths for train and val
train_dir = os.path.join(dataset_root, "train")
val_dir = os.path.join(dataset_root, "val")

# 🔹 Get class names from subfolders in the train directory
classes = sorted(os.listdir(train_dir))

# Create dictionary for YAML content
yaml_data = {
    "path": dataset_root.replace("\\", "/"),
    "train": "train",
    "val": "val",
    "names": {i: cls for i, cls in enumerate(classes)}
}

# 🔹 Output YAML file path
output_yaml = os.path.join(os.path.dirname(dataset_root), "yoga_poseimage.yaml")

# 🔹 Save YAML file
with open(output_yaml, "w") as f:
    yaml.dump(yaml_data, f, sort_keys=False)

print(f"✅ YAML file generated successfully at:\n{output_yaml}")
print("\nDetected classes:")
for i, cls in enumerate(classes):
    print(f"  {i}: {cls}")
