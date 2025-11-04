# =====================================================
# train_pose_cnn_lstm.py ✅ Enhanced Version
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
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Flatten
from tensorflow.keras.utils import to_categorical

# =====================================================
# 1️⃣ Load Dataset
# =====================================================
CSV_PATH = "outputs/augmented_angles.csv"
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"❌ File not found: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)
print(f"✅ Loaded: {CSV_PATH}")
print("Columns:", df.columns.tolist()[:10], "...")

# =====================================================
# 2️⃣ Prepare Features & Labels
# =====================================================
X = df.drop(["pose", "frame"], axis=1, errors="ignore").values.astype("float32")
y = df["pose"].values

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"🎯 Classes: {list(le.classes_)}")

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("✅ Features normalized")

# =====================================================
# 3️⃣ Reshape for CNN + LSTM
# =====================================================
TIMESTEPS = 10
FEATURES = X_scaled.shape[1]

usable_len = (X_scaled.shape[0] // TIMESTEPS) * TIMESTEPS
X_scaled = X_scaled[:usable_len]
y_encoded = y_encoded[:usable_len]

X_seq = X_scaled.reshape(-1, TIMESTEPS, FEATURES)
y_seq = y_encoded.reshape(-1, TIMESTEPS)[:, -1]

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
# 5️⃣ Build CNN + LSTM Hybrid Model
# =====================================================
model = Sequential([
    Conv1D(64, kernel_size=3, activation='relu', input_shape=(TIMESTEPS, FEATURES)),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),

    LSTM(128, return_sequences=False),
    Dense(64, activation='relu'),
    Dropout(0.3),

    Dense(len(le.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# =====================================================
# 6️⃣ Train Model
# =====================================================
EPOCHS = 60  # same for all models
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
plt.title("🌀 Confusion Matrix - CNN+LSTM Yoga Pose Classification")
plt.tight_layout()
plt.show()

# =====================================================
# 9️⃣ Save Model
# =====================================================
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "cnn_lstm_yoga_stable.h5")
model.save(MODEL_PATH)

print(f"\n✅ Model saved at: {MODEL_PATH}")
print("✅ CNN+LSTM Training complete and evaluated successfully!")
