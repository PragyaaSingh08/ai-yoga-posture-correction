import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt

# ===============================
# 1️⃣ Load MoveNet Keypoints Data
# ===============================
CSV_PATH = "outputs/movenet_keypoints.csv"

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"❌ File not found: {CSV_PATH}\nMake sure you’ve extracted MoveNet keypoints first!")

df = pd.read_csv(CSV_PATH)
print(f"✅ Loaded {len(df)} samples from {CSV_PATH}")

# ===============================
# 2️⃣ Prepare Features & Labels
# ===============================
X = df.drop(["pose", "frame"], axis=1, errors="ignore").values
y = df["pose"].values

le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = tf.keras.utils.to_categorical(y_encoded)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reshape data into sequences (for temporal info)
TIMESTEPS = 10
FEATURES = X_scaled.shape[1]

usable_len = (X_scaled.shape[0] // TIMESTEPS) * TIMESTEPS
X_scaled = X_scaled[:usable_len]
y_categorical = y_categorical[:usable_len]

X_seq = X_scaled.reshape(-1, TIMESTEPS, FEATURES)
y_seq = y_categorical.reshape(-1, TIMESTEPS, y_categorical.shape[1])[:, -1, :]

print(f"✅ Input shape: {X_seq.shape}")

# ===============================
# 3️⃣ Train-Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)
print(f"📊 Train: {X_train.shape}, Test: {X_test.shape}")

# ===============================
# 4️⃣ Build CNN + LSTM Model
# ===============================
model = Sequential([
    Conv1D(64, 3, activation='relu', input_shape=(TIMESTEPS, FEATURES)),
    MaxPooling1D(2),
    Dropout(0.3),

    LSTM(128, return_sequences=False),
    Dense(64, activation='relu'),
    Dropout(0.3),

    Dense(y_seq.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ===============================
# 5️⃣ Train Model
# ===============================
EPOCHS = 60
BATCH_SIZE = 16

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=8,
    restore_best_weights=True,
    verbose=1
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop],
    verbose=1
)

# ===============================
# 6️⃣ Evaluate
# ===============================
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"\n🎯 Final Test Accuracy: {test_acc:.4f}")

# ===============================
# 7️⃣ Save Model
# ===============================
os.makedirs("models", exist_ok=True)
model.save("models/movenet_cnn_lstm_yoga.h5")
print("✅ Model saved at: models/movenet_cnn_lstm_yoga.h5")

# ===============================
# 8️⃣ Plot Training Graphs
# ===============================
plt.figure(figsize=(8,5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title("📈 Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
