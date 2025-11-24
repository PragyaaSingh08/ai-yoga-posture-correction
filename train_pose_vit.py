# =====================================================
# train_pose_resnet.py ✅ Stable TensorFlow Version
# =====================================================

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf

# =====================================================
# 1️⃣ Load Dataset
# =====================================================
CSV_PATH = "outputs/pose_keypoints.csv"  # or "augmented_angles.csv" if using angle features
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"❌ File not found: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)
print(f"✅ Loaded dataset with {len(df)} samples")

# Features & labels
X = df.drop(["pose", "frame"], axis=1, errors="ignore").values.astype("float32")
y = df["pose"].values

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
num_classes = len(le.classes_)
print(f"🎯 Classes: {list(le.classes_)}")

# =====================================================
# 2️⃣ Convert Keypoints to Pseudo-Images
# =====================================================
def keypoints_to_image(sample):
    # Reshape flat keypoints to (17, 2) if 34 features exist
    if sample.shape[0] == 34:
        sample = sample.reshape(17, 2)
    img = np.zeros((96, 96, 3), dtype=np.float32)
    x = np.clip((sample[:, 0] / np.max(sample[:, 0] + 1e-6)) * 95, 0, 95).astype(int)
    y = np.clip((sample[:, 1] / np.max(sample[:, 1] + 1e-6)) * 95, 0, 95).astype(int)
    for i in range(len(x)):
        img[y[i], x[i]] = [1.0, 1.0, 1.0]  # white point
    img = cv2.GaussianBlur(img, (5, 5), 0)
    return img

import cv2
X_imgs = np.array([keypoints_to_image(x) for x in X])
print(f"✅ Converted to pseudo-images: {X_imgs.shape}")

# =====================================================
# 3️⃣ Train-Test Split
# =====================================================
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, test_idx in sss.split(X_imgs, y_encoded):
    X_train, X_test = X_imgs[train_idx], X_imgs[test_idx]
    y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes)
y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes)
print(f"📊 Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# =====================================================
# 4️⃣ Define CNN Backbone (ResNet50 or MobileNetV3)
# =====================================================
print("🔄 Loading ResNet50 backbone...")
base_model = tf.keras.applications.ResNet50(
    include_top=False,
    weights='imagenet',
    input_shape=(96, 96, 3),
    pooling='avg'
)

# 👇 To use MobileNetV3 instead, comment above & uncomment below
# base_model = tf.keras.applications.MobileNetV3Small(
#     include_top=False,
#     weights='imagenet',
#     input_shape=(96, 96, 3),
#     pooling='avg'
# )

# Freeze base model
base_model.trainable = False

# =====================================================
# 5️⃣ Add Classification Head
# =====================================================
x = base_model.output
x = tf.keras.layers.Dense(256, activation='relu')(x)
x = tf.keras.layers.Dropout(0.3)(x)
output = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

model = tf.keras.Model(inputs=base_model.input, outputs=output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# =====================================================
# 6️⃣ Train the Model
# =====================================================
EPOCHS = 30
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
# 8️⃣ Confusion Matrix & Report
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
plt.title("🌀 Confusion Matrix - ResNet50 Yoga Pose Classification")
plt.tight_layout()
plt.show()

# =====================================================
# 9️⃣ Save Model
# =====================================================
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "resnet50_yoga_pose.h5")
model.save(MODEL_PATH)

print(f"\n✅ Model saved at: {MODEL_PATH}")
print("✅ Training complete and evaluated successfully!")
