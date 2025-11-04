# ==========================================================
# Yoga Pose Correction with Angle Feedback (Image/Webcam)
# ==========================================================
import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO
import pyttsx3
import math
import argparse

# Initialize pose and drawing utils
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
engine = pyttsx3.init()

# Load YOLOv12 classification model
print("📦 Loading YOLOv12 classification model...")
model = YOLO("runs/classify/train5/weights/best.pt")
print("✅ Model loaded successfully!")

POSE_CLASSES = ['Tadasana', 'Vrikshasana', 'Trikonasana', 'Bhujangasana', 'Chaturanga Dandasana']

# ----------------------------------------------------------
# Helper: calculate angle between three points
# ----------------------------------------------------------
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# ----------------------------------------------------------
# Feedback rules for posture correction
# ----------------------------------------------------------
def give_feedback(landmarks, image):
    feedback = []
    h, w, _ = image.shape

    # Extract landmark points
    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w,
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h]
    elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * w,
             landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * h]
    wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x * w,
             landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y * h]
    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * w,
           landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * h]
    knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x * w,
            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y * h]
    ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * w,
             landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * h]

    # Calculate key angles
    elbow_angle = calculate_angle(shoulder, elbow, wrist)
    shoulder_angle = calculate_angle(hip, shoulder, elbow)
    knee_angle = calculate_angle(hip, knee, ankle)

    # Provide rule-based feedback
    if elbow_angle < 160:
        feedback.append("Straighten your arms.")
    if shoulder_angle < 40:
        feedback.append("Lift your arms higher.")
    if knee_angle < 160:
        feedback.append("Straighten your legs.")
    if not feedback:
        feedback.append("Good posture!")

    # Voice feedback
    engine.say(feedback[0])
    engine.runAndWait()

    return feedback[0]

# ----------------------------------------------------------
# Predict Yoga Pose + Give Feedback
# ----------------------------------------------------------
def process_image(image_path):
    image = cv2.imread(image_path)
    results = model(image)
    label = results[0].names[int(results[0].probs.top1)]
    conf = results[0].probs.top1conf.item() * 100

    with mp_pose.Pose(static_image_mode=True) as pose:
        result = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if result.pose_landmarks:
            mp_drawing.draw_landmarks(image, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            feedback = give_feedback(result.pose_landmarks.landmark, image)
        else:
            feedback = "Pose not detected."

    # Display output
    cv2.putText(image, f"{label} ({conf:.1f}%)", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)
    cv2.putText(image, feedback, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 3)
    cv2.imshow("Yoga Pose Correction", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ----------------------------------------------------------
# Webcam Mode
# ----------------------------------------------------------
def process_webcam():
    cap = cv2.VideoCapture(0)
    with mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            label = results[0].names[int(results[0].probs.top1)]
            conf = results[0].probs.top1conf.item() * 100

            result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if result.pose_landmarks:
                mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                feedback = give_feedback(result.pose_landmarks.landmark, frame)
            else:
                feedback = "Pose not detected."

            cv2.putText(frame, f"{label} ({conf:.1f}%)", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)
            cv2.putText(frame, feedback, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
            cv2.imshow("Yoga Pose Correction (Webcam)", frame)

            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

# ----------------------------------------------------------
# CLI Options
# ----------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, help="Path to input image")
    parser.add_argument("--webcam", action="store_true", help="Use webcam")
    args = parser.parse_args()

    if args.image:
        process_image(args.image)
    elif args.webcam:
        process_webcam()
    else:
        print("❌ Invalid mode! Use one of the following:")
        print("   ▶ python yoga_pose_feedback.py --image path/to/file.jpg")
        print("   ▶ python yoga_pose_feedback.py --webcam")
