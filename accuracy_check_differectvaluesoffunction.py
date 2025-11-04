# =====================================================
# train_and_evaluate_pose_models_fast.py
# ⚡ Fast Optimized Training Version
# =====================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

# =====================================================
# 1️⃣ Load Dataset
# =====================================================
CSV_PATH = "outputs/angles_pose_keypoints.csv"
df = pd.read_csv(CSV_PATH)
print(f"✅ Loaded: {CSV_PATH}")
print("Columns:", df.columns)

# Features and labels
feature_cols = ['left_knee', 'right_knee', 'left_elbow', 'right_elbow', 'left_shoulder', 'right_shoulder']
X = df[feature_cols].values
y = df['pose'].values

# Reshape for CNN/LSTM input
X = np.expand_dims(X, axis=1)
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
# 2️⃣ Define Lightweight Models
# =====================================================
def build_cnn_model(input_shape, num_classes):
    model = Sequential([
        Conv1D(32, 1, activation='relu', input_shape=input_shape),
        MaxPooling1D(1),
        Flatten(),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def build_lstm_model(input_shape, num_classes):
    model = Sequential([
        LSTM(64, input_shape=input_shape),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def build_cnn_lstm_model(input_shape, num_classes):
    model = Sequential([
        Conv1D(32, 1, activation='relu', input_shape=input_shape),
        MaxPooling1D(1),
        LSTM(64),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# =====================================================
# 3️⃣ Fast Training + Evaluation Function
# =====================================================
def train_and_evaluate(model, name):
    print(f"\n🚀 Training {name} model (fast mode)...")
    early_stop = EarlyStopping(monitor='val_accuracy', patience=2, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=8,
        batch_size=16,
        verbose=0,
        callbacks=[early_stop]
    )

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"✅ {name} Accuracy: {acc:.4f}")

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    y_true = np.argmax(y_test, axis=1)
    print(classification_report(y_true, y_pred, target_names=le.classes_))

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    return history, acc


# =====================================================
# 4️⃣ Train All Models Quickly
# =====================================================
input_shape = (X_train.shape[1], X_train.shape[2])
num_classes = y.shape[1]

cnn_model = build_cnn_model(input_shape, num_classes)
lstm_model = build_lstm_model(input_shape, num_classes)
hybrid_model = build_cnn_lstm_model(input_shape, num_classes)

cnn_hist, cnn_acc = train_and_evaluate(cnn_model, "CNN")
lstm_hist, lstm_acc = train_and_evaluate(lstm_model, "LSTM")
hybrid_hist, hybrid_acc = train_and_evaluate(hybrid_model, "CNN+LSTM Hybrid")


# =====================================================
# 5️⃣ Plot Accuracy Comparison
# =====================================================
def plot_training_curves(histories, title):
    plt.figure(figsize=(8, 4))
    for name, hist in histories.items():
        plt.plot(hist.history['val_accuracy'], label=f'{name} Val Acc')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Validation Accuracy')
    plt.legend()
    plt.show()

plot_training_curves(
    {"CNN": cnn_hist, "LSTM": lstm_hist, "Hybrid": hybrid_hist},
    "Pose Model Comparison (Fast)"
)


# =====================================================
# 6️⃣ Save Best Model Automatically
# =====================================================
best_model_name, best_acc = max(
    [("CNN", cnn_acc), ("LSTM", lstm_acc), ("Hybrid", hybrid_acc)],
    key=lambda x: x[1]
)

os.makedirs("outputs", exist_ok=True)
save_path = os.path.join("outputs", f"pose_{best_model_name.lower()}_fast.h5")

if best_model_name == "CNN":
    cnn_model.save(save_path)
elif best_model_name == "LSTM":
    lstm_model.save(save_path)
else:
    hybrid_model.save(save_path)

print(f"\n🏆 Best Model: {best_model_name} ({best_acc:.4f})")
print(f"💾 Saved to: {save_path}")
print("✅ Fast training complete!")
