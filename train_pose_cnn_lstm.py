import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import os

# === 1️⃣ Load data ===
INPUT_FILE = "outputs/augmented_angles.csv"
if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"❌ File not found: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)
print(f"✅ Loaded: {INPUT_FILE}")
print(f"Columns: {df.columns.tolist()[:10]} ...")

# === 2️⃣ Prepare features & labels ===
X = df.drop(["pose", "frame"], axis=1, errors="ignore").values
y = df["pose"].values

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === 3️⃣ Reshape for CNN + LSTM ===
# CNN expects [samples, timesteps, features]
TIMESTEPS = 10  # number of frames grouped together
FEATURES = X_scaled.shape[1]

# pad or trim samples to make it divisible by TIMESTEPS
usable_len = (X_scaled.shape[0] // TIMESTEPS) * TIMESTEPS
X_scaled = X_scaled[:usable_len]
y_categorical = y_categorical[:usable_len]

X_seq = X_scaled.reshape(-1, TIMESTEPS, FEATURES)
y_seq = y_categorical.reshape(-1, TIMESTEPS, y_categorical.shape[1])[:, -1, :]

print(f"✅ Reshaped to: {X_seq.shape}")

# === 4️⃣ Split data ===
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

# === 5️⃣ Build CNN + LSTM model ===
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

# === 6️⃣ Train ===
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=20,
    batch_size=32,
    verbose=1
)

# === 7️⃣ Save model ===
os.makedirs("models", exist_ok=True)
model.save("models/cnn_lstm_yoga.h5")
print("✅ Model saved at: models/cnn_lstm_yoga.h5")

# === 8️⃣ Evaluate ===
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"🎯 Test Accuracy: {test_acc:.4f}")
