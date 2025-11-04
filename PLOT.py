import matplotlib.pyplot as plt

models = ['LSTM', 'CNN+LSTM', 'CNN', 'GRU', 'MoveNet', 'Swin', 'VIT']
val_acc = [0.76, 0.79, 0.74, 0.70, 0.67, 0.55, .60]

plt.bar(models, val_acc, color=['red','orange','green','purple','blue','gray'])
plt.title('Model Validation Accuracy Comparison')
plt.ylabel('Validation Accuracy')
plt.ylim(0,5)
plt.show()
# ================================================================
# Yoga Pose Classification Training Script (YOLOv12n)
# ================================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO

# ================================================================
# STEP 1: Configuration
# ================================================================
MODEL_PATH = r"C:\Users\DR. Sindhu Sagar\Desktop\(79) WhatsApp_files\yoga posture\yolov12n-cls.pt"
DATASET_PATH = r"C:\Users\DR. Sindhu Sagar\Desktop\(79) WhatsApp_files\yoga posture\dataset"
RESULTS_DIR = r"C:\Users\DR. Sindhu Sagar\Desktop\(79) WhatsApp_files\yoga posture\runs\classify\YogaPose_Final"

# ================================================================
# STEP 2: Load and Train YOLOv12n Model
# ================================================================
print(f"🚀 Training on CPU using model: {MODEL_PATH}")

model = YOLO(MODEL_PATH)

results = model.train(
    data=DATASET_PATH,
    epochs=60,          # increase for better accuracy
    imgsz=224,
    lr0=0.001,
    batch=8,
    patience=10,
    optimizer="Adam",
    device="cpu",
    project=os.path.dirname(RESULTS_DIR),
    name=os.path.basename(RESULTS_DIR),
)

print("\n✅ Training complete! Results saved at:", RESULTS_DIR)

# ================================================================
# STEP 3: Load Training Results (results.csv)
# ================================================================
results_csv = os.path.join(RESULTS_DIR, "results.csv")
if not os.path.exists(results_csv):
    print("❌ ERROR: results.csv not found. Training might have failed.")
    exit()

df = pd.read_csv(results_csv)
print(f"✅ Loaded results from: {results_csv}")
print(f"📈 Columns available: {df.columns.tolist()}")

# ================================================================
# STEP 4: Plot Accuracy and Loss
# ================================================================
plt.figure(figsize=(10, 6))
plt.plot(df["epoch"], df["metrics/accuracy_top1"], label="Validation Accuracy (Top-1)", color="blue", linewidth=2)
plt.plot(df["epoch"], df["metrics/accuracy_top5"], label="Validation Accuracy (Top-5)", color="orange", linewidth=2)
plt.plot(df["epoch"], df["train/loss"], label="Training Loss", color="red", linestyle="--", linewidth=2)
plt.plot(df["epoch"], df["val/loss"], label="Validation Loss", color="green", linestyle="--", linewidth=2)

plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Value", fontsize=12)
plt.title("📊 YOLOv12n Yoga Pose Classification Training Progress", fontsize=14)
plt.legend()
plt.grid(True)
plt.tight_layout()

plot_path = os.path.join(RESULTS_DIR, "training_plot.png")
plt.savefig(plot_path, dpi=300)
print(f"📊 Saved training plot to: {plot_path}")

# ================================================================
# STEP 5: Extract Key Metrics
# ================================================================
final_epoch = df.iloc[-1]
train_loss = final_epoch.get("train/loss", None)
val_loss = final_epoch.get("val/loss", None)
val_acc_top1 = final_epoch.get("metrics/accuracy_top1", None)
val_acc_top5 = final_epoch.get("metrics/accuracy_top5", None)

# ================================================================
# STEP 6: Generate Remarks
# ================================================================
remarks = ""
if val_acc_top1 >= 0.9:
    remarks = "🔥 Excellent performance — near perfect classification."
elif val_acc_top1 >= 0.85:
    remarks = "✅ Very good model accuracy (above 85%). Suitable for deployment."
elif val_acc_top1 >= 0.7:
    remarks = "⚙️ Moderate performance. Consider increasing epochs or data augmentation."
else:
    remarks = "❗Low accuracy. Improve dataset balance or fine-tune hyperparameters."

# ================================================================
# STEP 7: Save Summary Report
# ================================================================
summary_path = os.path.join(RESULTS_DIR, "training_summary.txt")

with open(summary_path, "w", encoding="utf-8") as f:
    f.write("📘 Yoga Pose Classification Summary (YOLOv12n)\n")
    f.write("====================================================\n")
    f.write(f"Model Used: {MODEL_PATH}\n")
    f.write(f"Dataset: {DATASET_PATH}\n\n")
    f.write(f"Training Accuracy (Top-1): {val_acc_top1*100:.2f}%\n")
    f.write(f"Validation Accuracy (Top-5): {val_acc_top5*100:.2f}%\n")
    f.write(f"Training Loss: {train_loss:.4f}\n")
    f.write(f"Validation Loss: {val_loss:.4f}\n")
    f.write(f"\nRemarks: {remarks}\n")

print("\n✅ Summary saved at:", summary_path)

# ================================================================
# STEP 8: Final Output
# ================================================================
print("\n📊 Final Training Summary:")
print(f"Model: YOLOv12n")
print(f"Training Loss: {train_loss:.4f}")
print(f"Validation Loss: {val_loss:.4f}")
print(f"Top-1 Accuracy: {val_acc_top1*100:.2f}%")
print(f"Top-5 Accuracy: {val_acc_top5*100:.2f}%")
print(f"Remarks: {remarks}")
