# ================================================================
# 🧘 Real-Time Yoga Pose Classification (Offline YOLOv12n Version)
# Author: Pragya Singh
# ================================================================

import cv2
import torch
import pyttsx3
import time
import os
from ultralytics import YOLO

# ================================================================
# STEP 1: CONFIGURATION
# ================================================================
# ✅ Use your trained model (update path below if needed)
MODEL_PATH = r"C:\Users\DR. Sindhu Sagar\Desktop\(79) WhatsApp_files\yoga posture\runs\classify\train2\weights\best.pt"

# ✅ Prevent Ultralytics from connecting online
os.environ["YOLO_OFFLINE"] = "1"
os.environ["ULTRALYTICS_HUB_ENABLED"] = "0"
os.environ["WANDB_DISABLED"] = "true"
os.environ["NO_GITHUB"] = "1"

# 🔒 Check if model file exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model file not found: {MODEL_PATH}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Using device: {device}")

print("\n🔍 Loading YOLOv12n model (offline)...")
try:
    model = YOLO(MODEL_PATH)
    model.to(device)
    print("✅ Model loaded successfully (offline mode)!")
except Exception as e:
    raise RuntimeError(f"❌ Model load failed: {e}")

# ================================================================
# STEP 2: AUDIO FEEDBACK SETUP
# ================================================================
try:
    engine = pyttsx3.init()
    engine.setProperty('rate', 175)
    engine.setProperty('volume', 1.0)
    audio_enabled = True
except Exception:
    print("⚠️ Audio engine not available. Continuing silently.")
    audio_enabled = False


def speak_once(text):
    """Speak aloud if audio is available."""
    if audio_enabled:
        engine.say(text)
        engine.runAndWait()


# ================================================================
# STEP 3: CAMERA SETUP
# ================================================================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise Exception("❌ Cannot open webcam. Please check your camera connection.")

print("\n🎥 Starting Real-Time Yoga Pose Detection (Offline)...")
print("👉 Press 'q' to quit anytime.\n")

# ================================================================
# STEP 4: REAL-TIME PREDICTION LOOP
# ================================================================
last_feedback = ""
last_speech_time = 0
COOLDOWN = 3  # seconds between feedbacks

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame capture failed.")
        break

    try:
        # Predict using YOLO classification model
        results = model.predict(frame, imgsz=224, device=device, verbose=False)
        probs = results[0].probs if hasattr(results[0], "probs") else None

        if probs is not None:
            conf, cls_idx = torch.max(probs, dim=0)
            conf = float(conf)
            class_name = model.names[int(cls_idx)]

            # Color coding for confidence
            color = (0, 255, 0) if conf > 0.8 else (0, 255, 255) if conf > 0.4 else (0, 0, 255)
            label = f"{class_name}: {conf * 100:.1f}%"
            cv2.putText(frame, label, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

            # Feedback message
            if conf > 0.8:
                feedback = f"✅ Perfect {class_name.replace('_', ' ')}"
            elif conf > 0.4:
                feedback = f"⚠ Adjust your posture"
            else:
                feedback = f"❌ Pose not detected clearly"

            cv2.putText(frame, feedback, (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

            # Speak feedback with cooldown
            current_time = time.time()
            if feedback != last_feedback or (current_time - last_speech_time) > COOLDOWN:
                speak_once(feedback)
                last_feedback = feedback
                last_speech_time = current_time

        else:
            cv2.putText(frame, "❌ Pose not detected clearly", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            if last_feedback != "no_pose":
                speak_once("Pose not detected clearly")
                last_feedback = "no_pose"

    except Exception as e:
        print(f"❌ Prediction Error: {e}")
        cv2.putText(frame, "⚠ Prediction Error", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # Show camera feed
    cv2.imshow("🧘 Real-Time Yoga Pose Detection (Offline)", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ================================================================
# STEP 5: CLEANUP
# ================================================================
cap.release()
cv2.destroyAllWindows()
if audio_enabled:
    engine.stop()
print("\n✅ Yoga Pose Detection Ended. Namaste 🙏")
