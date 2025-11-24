# ==============================================================
# Yoga Pose Correction System (YOLOv12 + Mediapipe + Voice Guide)
# ==============================================================
import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO
import pyttsx3
import argparse
import math

# --------------------------------------------------------------
# Initialize mediapipe pose + drawing
# --------------------------------------------------------------
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
engine = pyttsx3.init()

# --------------------------------------------------------------
# Load YOLOv12 classification model
# --------------------------------------------------------------
print("📦 Loading YOLOv12 classification model...")
model = YOLO("runs/classify/train5/weights/best.pt")
print("✅ Model loaded successfully!")

# --------------------------------------------------------------
# Helper: calculate angle between three points
# --------------------------------------------------------------
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# --------------------------------------------------------------
# Ideal angle ranges for each yoga pose
# --------------------------------------------------------------
pose_angle_ranges = {
    "Tadasana": {"knee": (170, 180), "elbow": (160, 180)},
    "Utkatasana": {"knee": (80, 100), "hip": (60, 90)},
    "Bhujangasana": {"shoulder": (40, 70), "elbow": (160, 180)},
    "Trikonasana": {"hip": (150, 180), "shoulder": (160, 180)},
    "Vrikshasana": {"knee": (160, 180), "hip": (160, 180)},
}

# --------------------------------------------------------------
# Pose correction guidance text for each posture
# --------------------------------------------------------------
pose_corrections = {
    "Bhujangasana": "Lift your chest higher and keep elbows close to your ribs.",
    "Tadasana": "Straighten your spine, keep shoulders relaxed, and balance evenly on both feet.",
    "Vrikshasana": "Keep your gaze steady, open your chest, and maintain balance on one leg.",
    "Trikonasana": "Stretch both arms in one line and open your chest sideways.",
    "Utkatasana": "Bend your knees more, push hips back, and keep spine straight.",
}

# --------------------------------------------------------------
# Dynamic feedback based on detected pose + angles
# --------------------------------------------------------------
def give_pose_specific_feedback(landmarks, image, detected_pose):
    h, w, _ = image.shape
    feedback = []

    # Extract important landmarks
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

    # Calculate joint angles
    elbow_angle = calculate_angle(shoulder, elbow, wrist)
    shoulder_angle = calculate_angle(hip, shoulder, elbow)
    knee_angle = calculate_angle(hip, knee, ankle)

    # Fetch pose target ranges
    target = pose_angle_ranges.get(detected_pose, {})

    # Compare and add feedback dynamically
    if "knee" in target:
        low, high = target["knee"]
        if not (low <= knee_angle <= high):
            feedback.append("Adjust your knee angle.")
    if "elbow" in target:
        low, high = target["elbow"]
        if not (low <= elbow_angle <= high):
            feedback.append("Straighten or relax your arms.")
    if "shoulder" in target:
        low, high = target["shoulder"]
        if not (low <= shoulder_angle <= high):
            feedback.append("Lift or drop your shoulders slightly.")
    if "hip" in target:
        low, high = target["hip"]
        if not (low <= shoulder_angle <= high):
            feedback.append("Adjust your hip position for better balance.")

    # Add pose-specific instruction if needed
    if not feedback:
        feedback.append("Good posture for " + detected_pose + "!")
    else:
        feedback.append(pose_corrections.get(detected_pose, ""))

    # Voice feedback
    message = " ".join(feedback)
    engine.say(message)
    engine.runAndWait()

    return message

# --------------------------------------------------------------
# Process single image
# --------------------------------------------------------------
def process_image(image_path):
    image = cv2.imread(image_path)
    results = model(image)
    detected_pose = results[0].names[int(results[0].probs.top1)]
    conf = results[0].probs.top1conf.item() * 100

    with mp_pose.Pose(static_image_mode=True) as pose:
        result = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if result.pose_landmarks:
            mp_drawing.draw_landmarks(image, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            feedback = give_pose_specific_feedback(result.pose_landmarks.landmark, image, detected_pose)
        else:
            feedback = "Pose not detected."

    cv2.putText(image, f"{detected_pose} ({conf:.1f}%)", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    cv2.putText(image, feedback, (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.imshow("Yoga Pose Correction", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# --------------------------------------------------------------
# Webcam (real-time feedback)
# --------------------------------------------------------------
def process_webcam():
    cap = cv2.VideoCapture(0)
    with mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            detected_pose = results[0].names[int(results[0].probs.top1)]
            conf = results[0].probs.top1conf.item() * 100

            result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if result.pose_landmarks:
                mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                feedback = give_pose_specific_feedback(result.pose_landmarks.landmark, frame, detected_pose)
            else:
                feedback = "Pose not detected."

            cv2.putText(frame, f"{detected_pose} ({conf:.1f}%)", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            cv2.putText(frame, feedback, (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("Yoga Pose Correction (Webcam)", frame)

            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

# --------------------------------------------------------------
# CLI options
# --------------------------------------------------------------
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
        print(" Invalid mode! Use:")
        print("    python yoga_pose_feedback_dynamic.py --image path/to/image.jpg")
        print("    python yoga_pose_feedback_dynamic.py --webcam")
