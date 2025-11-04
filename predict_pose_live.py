# =====================================================
# predict_pose_live_cnn_lstm.py
# =====================================================

import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from collections import deque
import tensorflow as tf

# --- Configuration ---
MODEL_PATH = "outputs/pose_cnn_lstm_model.h5"   # ✅ your trained CNN+LSTM model
POSE_CLASSES = ['Bhujangasana', 'Tadasana', 'Vrikshasana', 'Trikonasana', 'WarriorPose']
TIMESTEPS = 16                                  # same as used in training

# --- Load trained model ---
model = load_model(MODEL_PATH)

# --- Initialize MediaPipe Pose ---
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# --- Helper: extract (x, y) coordinates of all 33 landmarks ---
def extract_keypoints(landmarks):
    keypoints = np.array([[lm.x, lm.y] for lm in landmarks], dtype=np.float32)
    keypoints = keypoints.flatten()  # 33 * 2 = 66 features
    return keypoints

# --- Store last TIMESTEPS frames ---
sequence = deque(maxlen=TIMESTEPS)

# --- Start webcam ---
cap = cv2.VideoCapture(0)  # or replace with path to video

with mp_pose.Pose(min_detection_confidence=0.6,
                  min_tracking_confidence=0.6) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip & preprocess
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        # --- If pose detected ---
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame,
                                      results.pose_landmarks,
                                      mp_pose.POSE_CONNECTIONS)

            keypoints = extract_keypoints(results.pose_landmarks.landmark)
            sequence.append(keypoints)

            # --- Predict when we have enough frames ---
            if len(sequence) == TIMESTEPS:
                X = np.expand_dims(sequence, axis=0)  # shape: (1, 16, 66)
                preds = model.predict(X, verbose=0)
                label_idx = np.argmax(preds)
                confidence = np.max(preds)
                label = f"{POSE_CLASSES[label_idx]} ({confidence*100:.1f}%)"

                # Overlay prediction
                cv2.rectangle(frame, (10, 10), (460, 70), (0, 0, 0), -1)
                cv2.putText(frame, label, (20, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        # --- Show frame ---
        cv2.imshow("🧘 Yoga Pose Detection (CNN+LSTM)", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
            break

cap.release()
cv2.destroyAllWindows()
