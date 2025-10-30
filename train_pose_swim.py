# =====================================================
# train_pose_swin.py — Final Stable Version
# Swin Transformer + Lightweight Classifier
# Author: Pragya Singh
# =====================================================

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import cv2

# =====================================================
# 1️⃣ Load CSV Dataset
# =====================================================
CSV_PATH = "outputs/angles_pose_keypoints.csv"
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"❌ CSV not found at {CSV_PATH}")

df = pd.read_csv(CSV_PATH)
print(f"✅ Loaded dataset with {len(df)} samples")

# Define features and labels
feature_cols = ['left_knee', 'right_knee', 'left_elbow',
                'right_elbow', 'left_shoulder', 'right_shoulder']
X = df[feature_cols].values.astype('float32')
y = df['pose'].values

# Encode labels
le = LabelEncoder()
y = le.fit_transform(y)
y = to_categorical(y)
print(f"🎯 Classes: {list(le.classes_)}")

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
print(f"📊 Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# =====================================================
# 2️⃣ Convert Numeric Features to Pseudo-Images
# =====================================================
def features_to_image(features):
    """
    Convert 6 numeric angles into a small 224x224 pseudo-image.
    """
    canvas = np.zeros((224, 224, 3), dtype=np.uint8)
    values = np.clip(features, 0, 180)
    normalized = (values / 180.0 * 200 + 12).astype(int)
    for i, v in enumerate(normalized):
        cv2.circle(canvas, (v, 40 * (i + 1)), 10, (255, 255, 255), -1)
    return canvas / 255.0

X_train_img = np.array([features_to_image(x) for x in X_train])
X_test_img = np.array([features_to_image(x) for x in X_test])
print(f"✅ Converted to pseudo-images: {X_train_img.shape}")

# =====================================================
# 3️⃣ Load Swin Transformer (Frozen Backbone)
# =====================================================
print("🔄 Loading Swin Transformer (Tiny) backbone...")
SWIN_URL = "https://tfhub.dev/sayakpaul/swin_tiny_patch4_window7_224/1"
swin_layer = hub.KerasLayer(SWIN_URL, trainable=False)

# =====================================================
# 4️⃣ Extract Swin Features (Handles 2D or 4D Output)
# =====================================================
def get_swin_features(images, batch_size=8):
    features = []
    total_batches = int(np.ceil(len(images) / batch_size))
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        outputs = swin_layer(batch)

        # Some models return dicts, others tensors
        if isinstance(outputs, dict):
            out = outputs['default']
        else:
            out = outputs

        # ✅ FIX: If it's 4D -> pool, if 2D -> already flattened
        if len(out.shape) == 4:
            out = tf.keras.layers.GlobalAveragePooling2D()(out)
        elif len(out.shape) == 2:
            out = out
        else:
            raise ValueError(f"Unexpected Swin output shape: {out.shape}")

        features.append(out)
        print(f"🔹 Processed batch {i//batch_size + 1}/{total_batches}")
    return tf.concat(features, axis=0)

print("🔍 Extracting Swin embeddings (this may take a few minutes)...")
X_train_embed = get_swin_features(X_train_img, batch_size=8)
X_test_embed = get_swin_features(X_test_img, batch_size=8)
print(f"✅ Swin embeddings: Train={X_train_embed.shape}, Test={X_test_embed.shape}")

# =====================================================
# 5️⃣ Train a Lightweight Classifier
# =====================================================
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train_embed.shape[1],)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(y_train.shape[1], activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# =====================================================
# 6️⃣ Train
# =====================================================
history = model.fit(
    X_train_embed, y_train,
    validation_data=(X_test_embed, y_test),
    epochs=20,
    batch_size=16,
    verbose=1
)

# =====================================================
# 7️⃣ Save Model
# =====================================================
os.makedirs("outputs", exist_ok=True)
MODEL_PATH = "outputs/pose_swin_model_final.h5"
model.save(MODEL_PATH)
print(f"✅ Model saved at: {MODEL_PATH}")
print("🎉 Training complete successfully!")
