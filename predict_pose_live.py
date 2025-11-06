# =============================================================
# predict_pose_with_verifier.py
# CNN+LSTM (or YOLO) prediction + MediaPipe geometric verifier
# Auto-corrects common misclassifications (e.g., Warrior <-> Tree)
# =============================================================

import os
import time
import math
import argparse
import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from tensorflow.keras.models import load_model
import pyttsx3

# ----------------- CONFIG -----------------
MODEL_PATH = "models/cnn_lstm_yoga_stable.h5"   # update if different
TIMESTEPS = 10                                  # same as training
POSE_CLASSES = ['Bhujangasana', 'Tadasana', 'Trikonasana', 'Vrikshasana', 'WarriorPose']
CONF_THRESHOLD = 0.55                           # classifier confidence threshold
VOICE_COOLDOWN = 3.0                            # seconds between spoken messages
LOG_CORRECTIONS = "pose_corrections.log"

# ----------------- Load model -----------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file missing: {MODEL_PATH}")
model = load_model(MODEL_PATH)

# ----------------- MediaPipe -----------------
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# ----------------- TTS -----------------
engine = pyttsx3.init()
engine.setProperty('rate', 155)
engine.setProperty('volume', 0.9)

# ----------------- Helpers -----------------
def calculate_angle(a, b, c):
    """Return angle (degrees) at point b formed by a-b-c."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    cosang = np.dot(ba, bc) / denom
    cosang = float(np.clip(cosang, -1.0, 1.0))
    ang = math.degrees(math.acos(cosang))
    return ang

def dist(a, b):
    return float(np.linalg.norm(np.array(a) - np.array(b)))

# Example keypoints extractor that matches a typical LSTM trained on angles:
def extract_angle_features(landmarks):
    """Return 6-angle feature vector per frame (adapt if your training used different features)."""
    lm = landmarks
    # left arm angle (shoulder-elbow-wrist)
    left_arm = calculate_angle([lm[11].x, lm[11].y],[lm[13].x, lm[13].y],[lm[15].x, lm[15].y])
    # right arm angle
    right_arm = calculate_angle([lm[12].x, lm[12].y],[lm[14].x, lm[14].y],[lm[16].x, lm[16].y])
    # left leg (hip-knee-ankle)
    left_leg = calculate_angle([lm[23].x, lm[23].y],[lm[25].x, lm[25].y],[lm[27].x, lm[27].y])
    # right leg
    right_leg = calculate_angle([lm[24].x, lm[24].y],[lm[26].x, lm[26].y],[lm[28].x, lm[28].y])
    # torso left side (shoulder-hip-knee)
    torso_left = calculate_angle([lm[11].x, lm[11].y],[lm[23].x, lm[23].y],[lm[25].x, lm[25].y])
    # torso right side
    torso_right = calculate_angle([lm[12].x, lm[12].y],[lm[24].x, lm[24].y],[lm[26].x, lm[26].y])
    return np.array([left_arm, right_arm, left_leg, right_leg, torso_left, torso_right], dtype=np.float32)

def detect_tree_pose_from_landmarks(landmarks, image_shape):
    """
    Heuristic check for Vrikshasana (Tree Pose).
    Returns True if geometry strongly suggests Tree Pose.
    """
    h, w = image_shape[:2]
    lm = landmarks

    # Landmarks indices for MediaPipe
    L_ANKLE = 27  # right ankle
    R_ANKLE = 28  # (note: mp indices may vary — verify for your version)
    L_KNEE = 25
    R_KNEE = 26
    L_HIP = 23
    R_HIP = 24
    L_SHOULDER = 11
    R_SHOULDER = 12
    L_WRIST = 15
    R_WRIST = 16

    # convert normalized coords to pixels (x,y)
    def pt(i):
        return np.array([landmarks[i].x * w, landmarks[i].y * h])

    # Determine which foot is lifted: check distances of each ankle to its own hip/knee
    left_ankle = pt(L_ANKLE)
    right_ankle = pt(R_ANKLE)
    left_knee = pt(L_KNEE)
    right_knee = pt(R_KNEE)
    left_hip = pt(L_HIP)
    right_hip = pt(R_HIP)

    # distances: ankle-to-opposite-thigh/knee (approx)
    d_leftankle_to_rightknee = dist(left_ankle, right_knee)
    d_leftankle_to_rightthigh = dist(left_ankle, right_hip)
    d_rightankle_to_leftknee = dist(right_ankle, left_knee)
    d_rightankle_to_leftthigh = dist(right_ankle, left_hip)

    # image size scale: use diagonal to normalize
    diag = math.hypot(w, h)
    # thresholds as fraction of diag
    close_thresh = 0.12 * diag   # small distance = foot near thigh/knee
    # check whether one ankle is close to opposite thigh/knee
    left_on_right_thigh = (d_leftankle_to_rightthigh < close_thresh) or (d_leftankle_to_rightknee < close_thresh)
    right_on_left_thigh = (d_rightankle_to_leftthigh < close_thresh) or (d_rightankle_to_leftknee < close_thresh)

    # check standing leg straightness by computing knee angle on standing leg
    # If left foot is on right thigh, the standing leg is right leg -> check right_leg angle
    # Use hip-knee-ankle angle (in pixels)
    right_leg_angle = calculate_angle(pt(R_HIP), pt(R_KNEE), pt(R_ANKLE))
    left_leg_angle = calculate_angle(pt(L_HIP), pt(L_KNEE), pt(L_ANKLE))

    # arms: check if both wrists are above shoulders (arms up)
    left_wrist = pt(L_WRIST)
    right_wrist = pt(R_WRIST)
    left_sh = pt(L_SHOULDER)
    right_sh = pt(R_SHOULDER)
    arms_up = (left_wrist[1] < left_sh[1]) and (right_wrist[1] < right_sh[1])

    # Decide
    # Conditions (tuned heuristics): one ankle close to opposite thigh/knee + standing leg approx straight + arms up
    tree_left = left_on_right_thigh and (right_leg_angle > 155) and arms_up
    tree_right = right_on_left_thigh and (left_leg_angle > 155) and arms_up

    return tree_left or tree_right

def detect_warrior_pose_from_landmarks(landmarks, image_shape):
    """Heuristic check for WarriorPose (front knee bent, back leg straight, torso forward)."""
    h, w = image_shape[:2]
    lm = landmarks

    def pt(i):
        return np.array([landmarks[i].x * w, landmarks[i].y * h])

    L_HIP, R_HIP = 23, 24
    L_KNEE, R_KNEE = 25, 26
    L_ANKLE, R_ANKLE = 27, 28
    L_SHOULDER, R_SHOULDER = 11, 12

    # compute knee angles
    left_leg_ang = calculate_angle(pt(L_HIP), pt(L_KNEE), pt(L_ANKLE))
    right_leg_ang = calculate_angle(pt(R_HIP), pt(R_KNEE), pt(R_ANKLE))

    # one knee should be bent around 60-120 deg and the other almost straight (>150)
    left_bent = (left_leg_ang < 140 and left_leg_ang > 60)
    right_bent = (right_leg_ang < 140 and right_leg_ang > 60)
    left_straight = (left_leg_ang > 150)
    right_straight = (right_leg_ang > 150)

    # arms roughly horizontal? compute shoulder-wrist vertical difference small relative to width -> approx horizontal
    def arm_horizontal():
        lw = np.array([landmarks[15].x * w, landmarks[15].y * h])
        rw = np.array([landmarks[16].x * w, landmarks[16].y * h])
        sh_left = np.array([landmarks[11].x * w, landmarks[11].y * h])
        sh_right = np.array([landmarks[12].x * w, landmarks[12].y * h])
        # average wrists y vs shoulders y
        wrists_y = (lw[1] + rw[1]) / 2.0
        shoulders_y = (sh_left[1] + sh_right[1]) / 2.0
        return abs(wrists_y - shoulders_y) < 0.18 * h

    arms_h = arm_horizontal()
    warrior = ((left_bent and right_straight) or (right_bent and left_straight)) and arms_h
    return warrior

# ----------------- LSTM feature extraction (angles) -----------------
def angles_frame_to_features(landmarks):
    # returns 6-angle vector (must match training)
    return extract_angle_features_for_model(landmarks)

# We will implement extract_angle_features_for_model similarly to earlier:
def extract_angle_features_for_model(landmarks):
    lm = landmarks
    # left arm
    left_arm = calculate_angle([lm[11].x, lm[11].y], [lm[13].x, lm[13].y], [lm[15].x, lm[15].y])
    # right arm
    right_arm = calculate_angle([lm[12].x, lm[12].y], [lm[14].x, lm[14].y], [lm[16].x, lm[16].y])
    # left leg
    left_leg = calculate_angle([lm[23].x, lm[23].y], [lm[25].x, lm[25].y], [lm[27].x, lm[27].y])
    # right leg
    right_leg = calculate_angle([lm[24].x, lm[24].y], [lm[26].x, lm[26].y], [lm[28].x, lm[28].y])
    # left torso
    torso_left = calculate_angle([lm[11].x, lm[11].y], [lm[23].x, lm[23].y], [lm[25].x, lm[25].y])
    # right torso
    torso_right = calculate_angle([lm[12].x, lm[12].y], [lm[24].x, lm[24].y], [lm[26].x, lm[26].y])
    return np.array([left_arm, right_arm, left_leg, right_leg, torso_left, torso_right], dtype=np.float32)

# ----------------- Spoken feedback control -----------------
last_spoken_time = 0.0

def speak_once(text):
    global last_spoken_time
    now = time.time()
    if now - last_spoken_time >= VOICE_COOLDOWN:
        engine.say(text)
        engine.runAndWait()
        last_spoken_time = now

# ----------------- Logging -----------------
def log_correction(orig_label, new_label, conf):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_CORRECTIONS, "a") as f:
        f.write(f"{ts} - {orig_label} ({conf:.2f}) -> {new_label}\n")

# ----------------- Prediction helper (image or frame) -----------------
from collections import deque as _deque
feature_seq = _deque(maxlen=TIMESTEPS)  # stores angle-vectors

def predict_from_frame_and_verify(frame, pose_results):
    """
    1) extract angles frame -> features
    2) append to seq; if seq full, run LSTM model
    3) get initial label + conf
    4) run rule-based verifier (e.g., detect_tree_pose)
    5) possibly override label
    """
    if not pose_results.pose_landmarks:
        return None, None, None

    lm = pose_results.pose_landmarks.landmark
    # compute angle features (6)
    feat = extract_angle_features_for_model(lm)
    feature_seq.append(feat)

    predicted_label = None
    confidence = None
    corrected = False

    if len(feature_seq) == TIMESTEPS:
        X = np.expand_dims(np.array(feature_seq), axis=0)  # shape (1, TIMESTEPS, 6)
        preds = model.predict(X, verbose=0)[0]
        idx = int(np.argmax(preds))
        predicted_label = POSE_CLASSES[idx]
        confidence = float(np.max(preds))

        # Run rule-based verifiers
        is_tree = detect_tree_pose_from_landmarks(lm, frame.shape)
        is_warrior = detect_warrior_pose_from_landmarks(lm, frame.shape)

        # Correction logic: if model says Warrior but landmarks indicate Tree strongly -> override
        if predicted_label == "WarriorPose" and is_tree:
            orig = predicted_label
            predicted_label = "Vrikshasana"  # Tree Pose
            corrected = True
            log_correction(orig, predicted_label, confidence)
        # conversely, if model says Vrikshasana but landmarks say warrior -> override
        elif predicted_label == "Vrikshasana" and is_warrior:
            orig = predicted_label
            predicted_label = "WarriorPose"
            corrected = True
            log_correction(orig, predicted_label, confidence)

    return predicted_label, confidence, corrected

# ----------------- UI overlays -----------------
def draw_overlay(frame, label, conf, corrected=False):
    h, w = frame.shape[:2]
    box_h = int(0.12*h)
    cv2.rectangle(frame, (0,0), (w, box_h), (0,0,0), -1)  # header
    color = (0,200,0) if not corrected else (0,255,255)
    text = f"{label} ({(conf*100) if conf is not None else 0:.1f}%)"
    cv2.putText(frame, text, (10, int(box_h*0.6)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    return frame

# ----------------- Main: image mode -----------------
def process_image(image_path):
    import imageio.v3 as iio
    if not os.path.exists(image_path):
        print("Image not found:", image_path); return
    img = iio.imread(image_path)
    frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.6) as pose:
        res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not res.pose_landmarks:
            print("No pose detected")
            cv2.imshow("frame", frame); cv2.waitKey(0); return
        label, conf, corrected = predict_from_frame_and_verify(frame, res)
        draw_overlay(frame, label if label else "Detecting...", conf if conf else 0.0, corrected)
        # speak
        tip = ""
        if label:
            if corrected:
                tip = f"Detected {label}. (Corrected from model)."
            else:
                tip = f"Detected {label}."
            speak_once(tip)
        cv2.imshow("Result", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# ----------------- Main: webcam mode -----------------
def process_webcam(cam_index=0):
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print("Cannot open webcam"); return
    with mp_pose.Pose(min_detection_confidence=0.55, min_tracking_confidence=0.55) as pose:
        print("Webcam started. Press ESC to quit.")
        while True:
            ok, frame = cap.read()
            if not ok: break
            frame = cv2.flip(frame, 1)
            res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            label, conf, corrected = predict_from_frame_and_verify(frame, res)
            if label:
                draw_overlay(frame, label, conf, corrected)
                if corrected:
                    speak_once(f"Model corrected to {label}")
                else:
                    # optionally speak only high-confidence initial predictions
                    if conf and conf >= 0.85:
                        speak_once(f"{label}")
            else:
                cv2.rectangle(frame, (0,0),(frame.shape[1], int(0.12*frame.shape[0])), (0,0,0), -1)
                cv2.putText(frame, "Detecting...", (10, int(0.07*frame.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)

            mp_drawing.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS) if res.pose_landmarks else None
            cv2.imshow("Pose Verifier (Press ESC to exit)", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    cap.release()
    cv2.destroyAllWindows()

# ----------------- CLI -----------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--image", type=str, help="Path to image")
    p.add_argument("--webcam", action="store_true", help="Run webcam")
    args = p.parse_args()
    if args.image:
        process_image(args.image)
    else:
        process_webcam()
