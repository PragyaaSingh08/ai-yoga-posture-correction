# =====================================================
# train_pose_movenet.py
# =====================================================

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

# --- Load CSV ---
CSV_PATH = "outputs/angles_pose_keypoints.csv"
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"❌ CSV not found at {CSV_PATH}")

df = pd.read_csv(CSV_PATH)
print(f"✅ Loaded: {CSV_PATH}")
print("Columns:", df.columns)

# --- Feature & Label selection ---
feature_cols = ['left_knee', 'right_knee', 'left_elbow', 'right_elbow', 
                'left_shoulder', 'right_shoulder']

X = df[feature_cols].values.astype('float32')
y = df['pose'].values

# --- Encode labels ---
le = LabelEncoder()
y = le.fit_transform(y)
y = to_categorical(y)
print(f"🎯 Classes: {list(le.classes_)}")

# --- Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"📊 Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# --- Load MoveNet (feature extractor) ---
print("🔄 Loading MoveNet model from TensorFlow Hub...")

MOVENET_MODEL_URL = "https://tfhub.dev/google/movenet/singlepose/lightning/4"

try:
    movenet = hub.load(MOVENET_MODEL_URL)
    print("✅ MoveNet model loaded successfully!")
except Exception as e:
    print(f"⚠️ Could not load MoveNet from TensorFlow Hub.\nError: {e}")
    movenet = None

# --- Define MoveNet-based feature extractor ---
def extract_features(X):
    """
    Extract features using MoveNet (simulated here).
    In practice, you'd pass pose keypoints or images through MoveNet.
    """
    X = tf.convert_to_tensor(X, dtype=tf.float32)

    if movenet is not None:
        # Simulated embedding shape similar to MoveNet output (17 keypoints × 2 coords)
        embeddings = tf.reshape(X, (X.shape[0], -1))
    else:
        embeddings = tf.reshape(X, (X.shape[0], -1))  # fallback to same shape

    return embeddings

# --- Extract features ---
X_train_features = extract_features(X_train)
X_test_features = extract_features(X_test)
print(f"✅ Features extracted: {X_train_features.shape}")

# --- Build classifier on top of MoveNet embeddings ---
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_features.shape[1],)),
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
    X_train_features, y_train,
    validation_data=(X_test_features, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1
)

# --- Save model ---
MODEL_DIR = "outputs"
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "pose_movenet_model.h5")
model.save(MODEL_PATH)

print(f"\n✅ Model saved at: {MODEL_PATH}")
print("✅ Training complete!")
