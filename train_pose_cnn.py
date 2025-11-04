# =====================================================
# train_pose_cnn.py ✅ Enhanced Version (Balanced + Evaluation)
# =====================================================

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# =====================================================
# 1️⃣ Load Dataset
# =====================================================
CSV_PATH = "outputs/angles_pose_keypoints.csv"
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"❌ CSV not found at {CSV_PATH}")

df = pd.read_csv(CSV_PATH)
print(f"✅ Loaded dataset with {len(df)} samples")
print("Columns:", df.columns)

# =====================================================
# 2️⃣ Feature & Label Setup
# =====================================================
feature_cols = ['left_knee', 'right_knee', 'left_elbow',
                'right_elbow', 'left_shoulder', 'right_shoulder']
X = df[feature_cols].values.astype("float32")
y = df['pose'].values

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)
print("✅ Features normalized using StandardScaler")

# Encode labels
le = LabelEncoder()
y = le.fit_transform(y)
print(f"🎯 Classes: {list(le.classes_)}")

# =====================================================
# 3️⃣ Reshape for CNN Input
# =====================================================
# CNN expects 3D input: (samples, timesteps/features, channels)
X = np.expand_dims(X, axis=2)
print(f"✅ Reshaped X for CNN: {X.shape}, y: {y.shape}")

# =====================================================
# 4️⃣ Stratified Train-Test Split
# =====================================================
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, test_idx in sss.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)
print(f"📊 Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# =====================================================
# 5️⃣ Build CNN Model
# =====================================================
model = Sequential([
    Conv1D(64, kernel_size=3, activation='relu', input_shape=(X.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(len(le.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# =====================================================
# 6️⃣ Train Model
# =====================================================
EPOCHS = 60  # same for all models for fair comparison
BATCH_SIZE = 16

history = model.fit(
    X_train, y_train_cat,
    validation_data=(X_test, y_test_cat),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1
)

# =====================================================
# 7️⃣ Evaluate Model
# =====================================================
train_loss, train_acc = model.evaluate(X_train, y_train_cat, verbose=0)
test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)

print("\n================== 📈 Model Performance ==================")
print(f"🏋️‍♀️ Training Accuracy: {train_acc:.4f}")
print(f"🧪 Test Accuracy: {test_acc:.4f}")
print("==========================================================\n")

# =====================================================
# 8️⃣ Classification Report & Confusion Matrix
# =====================================================
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

print("📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_,
            yticklabels=le.classes_)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("🌀 Confusion Matrix - CNN Yoga Pose Classification")
plt.tight_layout()
plt.show()

# =====================================================
# 9️⃣ Save Model
# =====================================================
MODEL_DIR = "outputs"
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "pose_cnn_model_stable.h5")
model.save(MODEL_PATH)

print(f"\n✅ Model saved at: {MODEL_PATH}")
print("✅ CNN Training complete and evaluated successfully!")
