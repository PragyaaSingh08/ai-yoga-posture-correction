import os

# Base dataset folder
base_dir = "dataset"

# Subfolders
splits = ["train", "val", "test"]
classes = ["downward_dog", "tree_pose", "cobra_pose", "warrior_pose"]

for split in splits:
    for cls in classes:
        os.makedirs(os.path.join(base_dir, split, cls), exist_ok=True)

print("✅ Folder structure created successfully!")
