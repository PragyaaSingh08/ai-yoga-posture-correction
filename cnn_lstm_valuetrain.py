# =====================================================
# evaluate_pose_cnn_lstm.py
# =====================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

# =====================================================
# 1️⃣ Load Dataset
# =====================================================
CSV_PATH = "outputs/angles_pose_keypoints.csv"
print("✅ Loading dataset...")
df = pd.read_csv(CSV_PATH)
print(f"✅ Dataset loaded successfully: {df.shape}")

# Feature columns (same as training)
feature_cols = ['left_knee', 'right_knee', 'left_elbow', 'right_elbow', 'left_shoulder', 'right_shoulder']

# Extract features and labels
X = df[feature_cols].values
y = df['pose'].values

# Reshape for CNN + LSTM input
X = np.expand_dims(X, axis=1)  # (samples, timesteps=1, features=6)
print(f"✅ Reshaped X: {X.shape}")

# Encode labels
le = LabelEncoder()
y = le.fit_transform(y)
y = to_categorical(y)
print(f"🎯 Classes: {list(le.classes_)}")

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"📊 Train: {X_train.shape}, Test: {X_test.shape}")

# =====================================================
# 2️⃣ Load Trained Model
# =====================================================
MODEL_PATH = "outputs/pose_cnn_lstm_hybrid_model.h5"
model = load_model(MODEL_PATH)
print(f"✅ Model loaded successfully from: {MODEL_PATH}")

# =====================================================
# 3️⃣ Evaluate Model
# =====================================================
print("🚀 Evaluating model performance...")
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

# =====================================================
# 4️⃣ Compute Evaluation Metrics
# =====================================================
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

print("\n📈 Model Evaluation Results:")
print(f"✅ Accuracy : {accuracy:.4f}")
print(f"✅ Precision: {precision:.4f}")
print(f"✅ Recall   : {recall:.4f}")
print(f"✅ F1-score : {f1:.4f}")

print("\n📊 Detailed Classification Report:")
print(classification_report(y_true, y_pred, target_names=le.classes_))

# =====================================================
# 5️⃣ Confusion Matrix Visualization & Saving
# =====================================================
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=le.classes_,
            yticklabels=le.classes_)
plt.title("🧩 Confusion Matrix - CNN+LSTM Pose Model")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()

# Save the figure
os.makedirs("outputs", exist_ok=True)
CM_PATH = "outputs/confusion_matrix.png"
plt.savefig(CM_PATH)
plt.close()

print(f"\n🖼️ Confusion matrix saved at: {CM_PATH}")
print("✅ Evaluation complete!")
