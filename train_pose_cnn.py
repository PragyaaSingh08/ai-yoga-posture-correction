# =====================================================
# train_pose_cnn.py
# =====================================================

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# --- Load CSV ---
CSV_PATH = "outputs/angles_pose_keypoints.csv"
df = pd.read_csv(CSV_PATH)
print(f"✅ Loaded: {CSV_PATH}")
print("Columns:", df.columns)

# --- Feature & Label selection ---
feature_cols = ['left_knee', 'right_knee', 'left_elbow', 'right_elbow', 'left_shoulder', 'right_shoulder']
X = df[feature_cols].values
y = df['pose'].values

# --- Reshape ---
X = np.expand_dims(X, axis=2)  # (samples, features, 1)
print(f"✅ Reshaped X: {X.shape}, y: {y.shape}")

# --- Encode labels ---
le = LabelEncoder()
y = le.fit_transform(y)
y = to_categorical(y)
print(f"🎯 Classes: {list(le.classes_)}")

# --- Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"📊 Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# --- Build CNN model ---
model = Sequential([
    Conv1D(64, kernel_size=3, activation='relu', input_shape=(X.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),
    Flatten(),
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

# --- Save model ---
MODEL_DIR = "outputs"
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "pose_cnn_model.h5")
model.save(MODEL_PATH)

print(f"\n✅ Model saved at: {MODEL_PATH}")
print("✅ CNN Training complete!")
