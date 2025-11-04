# =====================================================
# train_yolo_pose.py
# =====================================================
# Train YOLOv12 Classification model on yoga posture dataset
# Make sure yolov12n-cls.pt is in the same directory.
# =====================================================

import os
from ultralytics import YOLO

# 1️⃣ Load YOLOv12 classification model
model_path = "C:\Users\DR. Sindhu Sagar\Downloads\yolov12n-cls.pt"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Model weights not found: {model_path}")

model = YOLO(model_path)  # YOLOv12 classification model

# 2️⃣ Train the model
results = model.train(
    data="yoga_poseimage.yaml",  # Dataset config file
    epochs=50,                   # Number of epochs
    imgsz=224,                   # Image size for classification
    batch=16,                    # Batch size
    name="yoga_pose_yolov12",    # Run name (folder in runs/classify/)
    device='cpu'                 # Use 'cuda' if GPU available
)

# 3️⃣ Evaluate model performance (accuracy, confusion matrix, etc.)
metrics = model.val()
print("\n📊 Evaluation Metrics:")
print(metrics)

# 4️⃣ Test prediction on a sample image
test_image = r"dataset/val/TreePose/img1.jpg"  # Change this path
if os.path.exists(test_image):
    results = model.predict(
        source=test_image,
        save=True,
        show=True
    )
    print("\n✅ Prediction complete. Check 'runs/classify/predict/' for results.")
else:
    print("⚠️ Test image not found. Skipping prediction.")
