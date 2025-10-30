import os
import shutil
from sklearn.model_selection import train_test_split

# ✅ Path to your collected yoga video dataset
DATASET_DIR = "Yoga_vid_Collected"
OUTPUT_DIR = "dataset_split"  # all split data will go here

# ✅ Create folders for split data
for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(OUTPUT_DIR, split), exist_ok=True)

# ✅ Split ratios
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

# ✅ Collect all video file paths
all_videos = [os.path.join(DATASET_DIR, f) for f in os.listdir(DATASET_DIR)
              if f.lower().endswith(('.mp4', '.mov', '.avi', '.mkv'))]

print(f"Total videos found: {len(all_videos)}")

# ✅ Split into train, val, test
train_videos, temp_videos = train_test_split(all_videos, test_size=(1 - train_ratio), random_state=42)
val_videos, test_videos = train_test_split(temp_videos, test_size=(test_ratio / (test_ratio + val_ratio)), random_state=42)

def copy_files(video_list, split_name):
    for video_path in video_list:
        dest_folder = os.path.join(OUTPUT_DIR, split_name)
        shutil.copy(video_path, dest_folder)

# ✅ Copy to folders
copy_files(train_videos, 'train')
copy_files(val_videos, 'val')
copy_files(test_videos, 'test')

print("✅ Dataset split completed successfully!")
print(f"➡ Train: {len(train_videos)} videos")
print(f"➡ Val: {len(val_videos)} videos")
print(f"➡ Test: {len(test_videos)} videos")
print(f"📂 Split data saved inside: {os.path.abspath(OUTPUT_DIR)}")
