# ==========================================================
# Yoga Pose Correction System (YOLOv12 + MediaPipe + Voice)
# Author: Pragya Singh
# ==========================================================
import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO
import pyttsx3
import math
import argparse
import os

# ----------------------------------------------------------
# Initialize Mediapipe and Text-to-Speech
# ----------------------------------------------------------
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

engine = pyttsx3.init()
engine.setProperty('rate', 165)
engine.setProperty('volume', 0.9)

# ----------------------------------------------------------
# Load YOLOv12 Classification Model
# ----------------------------------------------------------
print("📦 Loading YOLOv12 classification model...")
model = YOLO("runs/classify/train5/weights/best.pt")
print("✅ Model loaded successfully!\n")

# Pose class labels (update according to your model)
POSE_CLASSES = [
    'Tadasana', 'Vrikshasana', 'Trikonasana',
    'Bhujangasana', 'Chaturanga Dandasana', 'Ardha Uttanasana',
    'Natarajasana', 'Utkatasana'
]

# ----------------------------------------------------------
# Helper: Calculate Angle Between Three Points
# ----------------------------------------------------------
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# ----------------------------------------------------------
# Pose-specific Feedback (visual + voice)
# ----------------------------------------------------------
def give_feedback(pose_name, landmarks, image):
    h, w, _ = image.shape
    l = mp_pose.PoseLandmark

    def pt(p): return [landmarks[p.value].x * w, landmarks[p.value].y * h]

    shoulder = pt(l.LEFT_SHOULDER)
    elbow = pt(l.LEFT_ELBOW)
    wrist = pt(l.LEFT_WRIST)
    hip = pt(l.LEFT_HIP)
    knee = pt(l.LEFT_KNEE)
    ankle = pt(l.LEFT_ANKLE)

    # Angles
    elbow_angle = calculate_angle(shoulder, elbow, wrist)
    shoulder_angle = calculate_angle(hip, shoulder, elbow)
    knee_angle = calculate_angle(hip, knee, ankle)
    back_angle = calculate_angle(shoulder, hip, knee)

    feedback = "Good posture!"
    voice_note = f"Good posture for {pose_name}!"

    # -----------------------------
    # Pose-specific correction logic
    # -----------------------------
    pname = pose_name.lower()

    if pname == "tadasana":
        if knee_angle < 170:
            feedback = "Straighten your legs!"
            voice_note = "Engage your thighs and press your feet evenly on the mat."

    elif pname == "vrikshasana":
        if knee_angle < 90:
            feedback = "Lift your leg higher!"
            voice_note = "Place your lifted foot on the inner thigh and keep balance."

    elif pname == "trikonasana":
        if back_angle < 140:
            feedback = "Open your chest more!"
            voice_note = "Rotate your chest upward and keep both arms aligned."

    elif pname == "bhujangasana":
        if shoulder_angle < 30:
            feedback = "Lift your chest higher!"
            voice_note = "Use your back muscles to lift your chest upward."

    elif pname == "chaturanga dandasana":
        if elbow_angle > 100:
            feedback = "Bend your elbows closer to 90°!"
            voice_note = "Keep elbows close to ribs and body in a straight line."

    elif pname == "ardha uttanasana":
        if back_angle < 150:
            feedback = "Straighten your back!"
            voice_note = "Keep your spine long and look slightly forward."

    elif pname == "natarajasana":
        if shoulder_angle < 40:
            feedback = "Lift your back leg higher!"
            voice_note = "Pull your lifted leg up and open your chest forward."

    elif pname == "utkatasana":
        if knee_angle > 120:
            feedback = "Bend your knees deeper!"
            voice_note = "Sit lower into your imaginary chair and keep spine straight."

    # -----------------------------
    # Voice feedback (only if confident)
    # -----------------------------
    engine.say(f"{feedback}. {voice_note}")
    engine.runAndWait()

    return feedback

# ----------------------------------------------------------
# Process Single Image
# ----------------------------------------------------------
def process_image(image_path):
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return

    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Unable to load image at {image_path}")
        return

    results = model(image)
    top_idx = int(results[0].probs.top1)
    label = results[0].names[top_idx]
    conf = results[0].probs.top1conf.item() * 100

    print(f"Predicted Pose: {label} ({conf:.2f}%)")

    # Process pose landmarks
    with mp_pose.Pose(static_image_mode=True) as pose:
        result = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if result.pose_landmarks:
            mp_drawing.draw_landmarks(image, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            feedback = give_feedback(label, result.pose_landmarks.landmark, image)
        else:
            feedback = "Pose not detected."

    # Overlay text
    cv2.putText(image, f"{label} ({conf:.1f}%)", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    cv2.putText(image, feedback, (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)

    cv2.imshow("Yoga Pose Correction", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ----------------------------------------------------------
# Webcam (Real-time Feedback)
# ----------------------------------------------------------
def process_webcam():
    cap = cv2.VideoCapture(0)
    with mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            top_idx = int(results[0].probs.top1)
            label = results[0].names[top_idx]
            conf = results[0].probs.top1conf.item() * 100

            result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if result.pose_landmarks:
                mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                feedback = give_feedback(label, result.pose_landmarks.landmark, frame)
            else:
                feedback = "Pose not detected."

            cv2.putText(frame, f"{label} ({conf:.1f}%)", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            cv2.putText(frame, feedback, (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow("Yoga Pose Correction (Webcam)", frame)

            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

# ----------------------------------------------------------
# Run the System
# ----------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Yoga Pose Correction using YOLOv12 + MediaPipe")
    parser.add_argument("--image", type=str, help="Path to input image")
    parser.add_argument("--webcam", action="store_true", help="Use webcam for live prediction")
    args = parser.parse_args()

    if args.image:
        process_image(args.image)
    elif args.webcam:
        process_webcam()
    else:
        print("❌ Please provide an input source:")
        print("   ▶ python PREDICT_YOLO.py --image path/to/image.jpg")
        print("   ▶ python PREDICT_YOLO.py --webcam")
# ================================================================
# Real-Time Yoga Pose Detection with Audio Feedback using YOLOv12
# Enhanced Version with AVIF Support
# Author: Pragya Singh
# ================================================================

import os
import cv2
import torch
import pyttsx3
import argparse
import imageio.v3 as iio
from ultralytics import YOLO
import numpy as np

# ------------------------------------------------------------
# Pose correction tips dictionary
# ------------------------------------------------------------
pose_corrections = {
    "Bhujangasana": "Lift your chest higher and keep your elbows close to the body.",
    "Tadasana": "Straighten your spine and balance your weight evenly on both feet.",
    "Vrikshasana": "Keep your gaze steady and palms together above the head.",
    "Trikonasana": "Open your chest and align both arms in a straight line.",
    "Utkatasana": "Bend your knees more and keep your spine straight.",
    "Ardha Uttanasana": "Straighten your back and lengthen your spine forward.",
    "Natarajasana": "Lift your back leg higher and open your chest forward."
}

# ------------------------------------------------------------
# Initialize voice feedback engine
# ------------------------------------------------------------
engine = pyttsx3.init()
engine.setProperty('rate', 165)  # Speech rate
engine.setProperty('volume', 0.9)  # Volume

def speak_feedback(text):
    """Speaks the correction aloud."""
    engine.say(text)
    engine.runAndWait()

# ------------------------------------------------------------
# Load YOLOv12 model
# ------------------------------------------------------------
print("📦 Loading YOLOv12 classification model...")
MODEL_PATH = "yoga_yolov12.pt"

try:
    model = YOLO(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading YOLOv12 model: {e}")
    exit()

# ------------------------------------------------------------
# Function to load image (supports .avif, .jpg, .png, etc.)
# ------------------------------------------------------------
def load_image(image_path):
    """Loads image safely using OpenCV or ImageIO."""
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return None

    image = cv2.imread(image_path)
    if image is None and image_path.lower().endswith(".avif"):
        try:
            image = iio.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"❌ Failed to read AVIF image: {e}")
            return None
    return image

# ------------------------------------------------------------
# Pose prediction and correction feedback
# ------------------------------------------------------------
def predict_pose(image):
    """Predict yoga pose and provide feedback."""
    results = model(image)
    pred = results[0].probs.top1  # Top predicted index
    conf = results[0].probs.top1conf.item()
    class_name = results[0].names[pred]

    print(f"\n🧘‍♀️ Predicted Pose: {class_name} ({conf*100:.2f}% confidence)")

    correction = pose_corrections.get(
        class_name, "Maintain balance and correct your alignment."
    )
    print(f"💡 Correction Tip: {correction}")
    speak_feedback(f"{class_name} detected. {correction}")

    return class_name, correction, conf

# ------------------------------------------------------------
# Process static image
# ------------------------------------------------------------
def process_image(image_path):
    image = load_image(image_path)
    if image is None:
        print("❌ Unable to load image.")
        return

    predict_pose(image)
    cv2.imshow("Yoga Pose", image)
    print("Press any key to exit...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ------------------------------------------------------------
# Real-time webcam detection
# ------------------------------------------------------------
def process_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Unable to access webcam.")
        return

    print("🎥 Webcam started... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        annotated_frame = results[0].plot()

        pred = results[0].probs.top1
        conf = results[0].probs.top1conf.item()
        class_name = results[0].names[pred]

        cv2.putText(annotated_frame, f"{class_name} ({conf*100:.1f}%)",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        if conf > 0.7:
            correction = pose_corrections.get(class_name, "Maintain posture alignment.")
            cv2.putText(annotated_frame, correction, (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            speak_feedback(f"{class_name} detected. {correction}")

        cv2.imshow("Real-Time Yoga Pose Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, help="Path to input image")
    parser.add_argument("--webcam", action="store_true", help="Use webcam for real-time detection")
    args = parser.parse_args()

    if args.image:
        process_image(args.image)
    elif args.webcam:
        process_webcam()
    else:
        print("⚠️ Please provide --image <path> or --webcam argument.")
