# =====================================================
# train_pose_gru.py
# =====================================================

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# --- Load CSV ---
CSV_PATH = "outputs/angles_pose_keypoints.csv"
df = pd.read_csv(CSV_PATH)
print(f"✅ Loaded: {CSV_PATH}")
print("Columns:", df.columns)

# --- Feature & Label selection ---
feature_cols = ['left_knee', 'right_knee', 'left_elbow', 'right_elbow', 'left_shoulder', 'right_shoulder']
X = df[feature_cols].values
y = df['pose'].values

# --- Parameters ---
TIMESTEPS = 20
FEATURES = len(feature_cols)

# --- Reshape safely ---
total = len(X)
usable = (total // TIMESTEPS) * TIMESTEPS
X = X[:usable]
y = y[:usable]

X = X.reshape(-1, TIMESTEPS, FEATURES)
print(f"✅ Reshaped X: {X.shape}, y: {y.shape}")

# --- Encode labels ---
le = LabelEncoder()
y = le.fit_transform(y)
y = y[:X.shape[0]]
y = to_categorical(y)
print(f"🎯 Classes: {list(le.classes_)}")

# --- Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"📊 Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# --- Build GRU model ---
model = Sequential([
    GRU(128, input_shape=(TIMESTEPS, FEATURES), return_sequences=False),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(y.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# --- Train ---
EPOCHS = 25
BATCH_SIZE = 16

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1
)

# --- Save ---
MODEL_DIR = "outputs"
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "pose_gru_model.h5")
model.save(MODEL_PATH)

print(f"\n✅ Model saved at: {MODEL_PATH}")
print("✅ GRU Training complete!")
