# =========================================
# train_model.py
# Train a yoga posture classifier (multi-class)
# using extracted joint angles + show accuracy/loss graphs
# =========================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# -----------------------------
# 1️⃣ Load full dataset CSV
# -----------------------------
CSV_PATH = "outputs/full_dataset_angles.csv"

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError("❌ Dataset not found! Run build_dataset.py first.")

df = pd.read_csv(CSV_PATH)
print(f"✅ Loaded dataset with {len(df)} samples")
print(f"📊 Columns: {df.columns.tolist()}")

# -----------------------------
# 2️⃣ Clean up and verify
# -----------------------------
df = df.dropna()

feature_cols = [
    'left_knee', 'right_knee',
    'left_elbow', 'right_elbow',
    'left_shoulder', 'right_shoulder'
]

if not set(feature_cols).issubset(df.columns):
    raise ValueError("❌ Missing some angle columns in CSV. Run extract_angles or build_dataset.py again.")

if 'label' not in df.columns:
    raise ValueError("❌ Missing 'label' column. Ensure dataset folder has subfolders as pose labels.")

# -----------------------------
# 3️⃣ Prepare X and y
# -----------------------------
X = df[feature_cols].values
y = df['label'].values

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
label_names = le.classes_
print(f"🧩 Detected {len(label_names)} classes: {label_names.tolist()}")

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# -----------------------------
# 4️⃣ Build neural network
# -----------------------------
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(len(label_names), activation='softmax')  # multi-class output
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# -----------------------------
# 5️⃣ Train the model
# -----------------------------
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=80,
    batch_size=16,
    callbacks=[early_stop],
    verbose=1
)

# -----------------------------
# 6️⃣ Plot training performance
# -----------------------------
plt.figure(figsize=(10, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig("outputs/training_performance.png")
plt.show()

# -----------------------------
# 7️⃣ Save model, scaler, and labels
# -----------------------------
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

model_path = os.path.join(MODEL_DIR, "yoga_pose_mlp_multiclass.h5")
scaler_path = os.path.join(MODEL_DIR, "scaler.npy")
labels_path = os.path.join(MODEL_DIR, "labels.npy")

model.save(model_path)
np.save(scaler_path, scaler.scale_)
np.save(labels_path, label_names)

print("\n✅ Training complete!")
print(f"🧠 Model saved at: {model_path}")
print(f"📈 Scaler saved at: {scaler_path}")
print(f"🏷️ Labels saved at: {labels_path}")
print("📊 Training graph saved at: outputs/training_performance.png")
