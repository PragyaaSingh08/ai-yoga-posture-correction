# =====================================================
# train_pose_gru.py ✅ Improved Version (Full Evaluation + Balanced)
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
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# =====================================================
# 1️⃣ Load CSV Dataset
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

# Normalize the feature data
scaler = StandardScaler()
X = scaler.fit_transform(X)
print("✅ Features normalized using StandardScaler")

# Encode the labels
le = LabelEncoder()
y = le.fit_transform(y)
print(f"🎯 Classes: {list(le.classes_)}")

# =====================================================
# 3️⃣ Prepare Sequence Data for GRU
# =====================================================
TIMESTEPS = 10  # reduced from 20 for smaller datasets
FEATURES = len(feature_cols)

usable = (len(X) // TIMESTEPS) * TIMESTEPS
X = X[:usable]
y = y[:usable]

X_seq = X.reshape(-1, TIMESTEPS, FEATURES)
y_seq = y.reshape(-1, TIMESTEPS)
y_seq = y_seq[:, -1]  # use label of last timestep

print(f"✅ Reshaped X: {X_seq.shape}, y: {y_seq.shape}")

# =====================================================
# 4️⃣ Stratified Train-Test Split
# =====================================================
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, test_idx in sss.split(X_seq, y_seq):
    X_train, X_test = X_seq[train_idx], X_seq[test_idx]
    y_train, y_test = y_seq[train_idx], y_seq[test_idx]

y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)
print(f"📊 Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# =====================================================
# 5️⃣ Build GRU Model
# =====================================================
model = Sequential([
    GRU(128, input_shape=(TIMESTEPS, FEATURES), return_sequences=False),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(len(le.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# =====================================================
# 6️⃣ Train Model
# =====================================================
EPOCHS = 60  # Same for all models for fair comparison
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
# 8️⃣ Classification Metrics
# =====================================================
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

print("📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_,
            yticklabels=le.classes_)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("🌀 Confusion Matrix - GRU Yoga Pose Classification")
plt.tight_layout()
plt.show()

# =====================================================
# 9️⃣ Save Model
# =====================================================
MODEL_DIR = "outputs"
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "pose_gru_model_stable.h5")
model.save(MODEL_PATH)
print(f"\n✅ Model saved at: {MODEL_PATH}")
print("✅ GRU Training complete and evaluated successfully!")
