# =====================================================
# train_pose_cnn_lstm_hybrid.py
# =====================================================

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Flatten
from tensorflow.keras.utils import to_categorical

# =====================================================
# 1️⃣ Load Dataset
# =====================================================
CSV_PATH = "outputs/angles_pose_keypoints.csv"
df = pd.read_csv(CSV_PATH)
print(f"✅ Loaded: {CSV_PATH}")
print("Columns:", df.columns)

# Select features and labels
feature_cols = ['left_knee', 'right_knee', 'left_elbow', 'right_elbow', 'left_shoulder', 'right_shoulder']
X = df[feature_cols].values
y = df['pose'].values

# Reshape for CNN+LSTM (samples, timesteps, features)
# We treat each keypoint as a "time step"
X = np.expand_dims(X, axis=1)  # shape -> (samples, 1, features)
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
# 2️⃣ Define CNN + LSTM Hybrid Model
# =====================================================
model = Sequential([
    # CNN feature extraction
    Conv1D(64, kernel_size=1, activation='relu', input_shape=(X.shape[1], X.shape[2])),
    MaxPooling1D(pool_size=1),
    Dropout(0.3),

    # Temporal pattern learning
    LSTM(128, return_sequences=False),
    Dropout(0.3),

    # Classification layers
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(y.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# =====================================================
# 3️⃣ Train Model
# =====================================================
EPOCHS = 25
BATCH_SIZE = 16

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1
)

# =====================================================
# 4️⃣ Save Model
# =====================================================
MODEL_DIR = "outputs"
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "pose_cnn_lstm_hybrid_model.h5")
model.save(MODEL_PATH)

print(f"\n✅ Model saved at: {MODEL_PATH}")
print("✅ CNN + LSTM Hybrid Training complete!")
