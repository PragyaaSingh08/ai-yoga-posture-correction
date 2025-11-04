# =====================================================
# train_pose_vit_fast.py — Yoga Pose Classification using Vision Transformer (ViT)
# =====================================================

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

# =====================================================
# ✅ Mixed Precision + XLA Optimization for Speed
# =====================================================
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')
tf.config.optimizer.set_jit(True)  # XLA enabled

# =====================================================
# 1️⃣ Load Dataset
# =====================================================
CSV_PATH = "outputs/angles_pose_keypoints.csv"
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"❌ CSV not found at {CSV_PATH}")

df = pd.read_csv(CSV_PATH)
print(f"✅ Loaded dataset with {len(df)} samples")

# =====================================================
# 2️⃣ Feature & Label Setup
# =====================================================
feature_cols = ['left_knee', 'right_knee', 'left_elbow',
                'right_elbow', 'left_shoulder', 'right_shoulder']
X = df[feature_cols].values.astype("float32")
y = df['pose'].values

# Encode labels
le = LabelEncoder()
y = le.fit_transform(y)
y = to_categorical(y)
print(f"🎯 Classes: {list(le.classes_)}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
print(f"📊 Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# =====================================================
# 3️⃣ Convert feature vectors → pseudo-images
# =====================================================
def to_pseudo_images(X, image_size=(96, 96, 3)):
    n = X.shape[0]
    pseudo = np.repeat(X[:, None, None, :], image_size[0], axis=1)
    pseudo = np.repeat(pseudo, image_size[1], axis=2)
    pseudo = pseudo[:, :, :, :3]
    pseudo = np.clip(pseudo / np.max(pseudo), 0, 1)
    return pseudo.astype("float32")

X_train_img = to_pseudo_images(X_train)
X_test_img = to_pseudo_images(X_test)
print(f"✅ Converted to pseudo-images: {X_train_img.shape}")

# =====================================================
# 4️⃣ tf.data Pipeline (faster loading)
# =====================================================
BATCH_SIZE = 32
train_ds = tf.data.Dataset.from_tensor_slices((X_train_img, y_train)).shuffle(512).batch(BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE)
test_ds = tf.data.Dataset.from_tensor_slices((X_test_img, y_test)).batch(BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE)

# =====================================================
# 5️⃣ Vision Transformer (ViT-B16)
# =====================================================
print("🔄 Loading Vision Transformer (ViT-B16)...")

vit_model = tf.keras.applications.ViT_B16(
    include_top=False,
    pooling='avg',
    input_shape=(96, 96, 3),
    weights='imagenet'
)

print("✅ Loaded ViT-B16 Backbone")

# =====================================================
# 6️⃣ Build Model
# =====================================================
inputs = tf.keras.Input(shape=(96, 96, 3))
x = vit_model(inputs, training=False)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
outputs = Dense(y.shape[1], activation='softmax', dtype='float32')(x)

model = Model(inputs, outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# =====================================================
# 7️⃣ Train Model
# =====================================================
EPOCHS = 60

history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=EPOCHS,
    verbose=1
)

# =====================================================
# 8️⃣ Evaluate Model
# =====================================================
train_loss, train_acc = model.evaluate(train_ds, verbose=0)
test_loss, test_acc = model.evaluate(test_ds, verbose=0)

print("\n================== 📈 Model Performance ==================")
print(f"🏋️‍♀️ Training Accuracy: {train_acc:.4f}")
print(f"🧪 Test Accuracy: {test_acc:.4f}")
print(f"⚙️  Backbone Used: Vision Transformer (ViT-B16, 96x96 input)")
print("==========================================================\n")

# =====================================================
# 9️⃣ Accuracy/Loss Graphs
# =====================================================
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val', linestyle='--')
plt.title('📊 Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val', linestyle='--')
plt.title('📉 Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()

# =====================================================
# 🔟 Classification Report & Confusion Matrix
# =====================================================
y_pred_probs = model.predict(test_ds)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

print("📊 Classification Report:")
print(classification_report(y_true, y_pred, target_names=le.classes_))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("⚡ Confusion Matrix — Yoga Pose Classification (ViT-B16)")
plt.tight_layout()
plt.show()

# =====================================================
# 1️⃣1️⃣ Save Model
# =====================================================
MODEL_DIR = "outputs"
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "pose_vit_fast.h5")
model.save(MODEL_PATH)
print(f"\n✅ Model saved at: {MODEL_PATH}")
print("✅ ViT Training Complete ⚡")
