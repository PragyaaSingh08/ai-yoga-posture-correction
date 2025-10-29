# =========================================
# YOLOv8 Pose Detection API (Video Upload)
# Author: Pragya Singh
# =========================================

from flask import Flask, request, jsonify, send_from_directory
from ultralytics import YOLO
import cv2
import os
import csv
import numpy as np
import uuid
from concurrent.futures import ThreadPoolExecutor
import torch

app = Flask(__name__)

# Automatically select device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on {device}")

# Load YOLOv8 Pose model once at startup
model = YOLO("yolov8n-pose.pt").to(device)

# Create required folders
os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs/frames", exist_ok=True)
os.makedirs("outputs/videos", exist_ok=True)

# Thread pool for parallel saving
executor = ThreadPoolExecutor(max_workers=4)


# -----------------------------
# Video processing function
# -----------------------------
def process_video(video_path, session_id):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Could not open video file.")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fps = fps if fps > 0 else 25  # fallback if FPS is 0

    output_video_path = f"outputs/videos/{session_id}_output.avi"
    out = cv2.VideoWriter(
        output_video_path,
        cv2.VideoWriter_fourcc(*'MJPG'),
        fps,
        (frame_width, frame_height)
    )

    csv_path = f"outputs/{session_id}_keypoints.csv"
    headers = ["frame"] + [f"x{i}" for i in range(17)] + [f"y{i}" for i in range(17)]
    csv_rows = []

    frame_count = 0
    frame_urls = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # Run YOLO pose detection
        results = model(frame, verbose=False)
        annotated_frame = results[0].plot()

        # Get keypoints safely
        kp = results[0].keypoints
        if kp is not None and hasattr(kp, "xy"):
            kp_array = kp.xy.cpu().numpy()
            for i, person in enumerate(kp_array):
                row = [frame_count] + [float(x) for x, y in person] + [float(y) for x, y in person]
                csv_rows.append(row)

        # Save annotated frame to video
        out.write(annotated_frame)

        # Save cropped humans in parallel
        boxes = results[0].boxes.xyxy.cpu().numpy()
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box[:4])
            cropped = frame[y1:y2, x1:x2]
            if cropped.size > 0:
                frame_name = f"{session_id}_frame_{frame_count}_{i}.jpg"
                frame_path = os.path.join("outputs/frames", frame_name)
                executor.submit(cv2.imwrite, frame_path, cropped)
                frame_urls.append(f"http://127.0.0.1:5000/frames/{frame_name}")

    cap.release()
    out.release()

    # Write CSV
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(csv_rows)

    return (
        f"http://127.0.0.1:5000/videos/{session_id}_output.avi",
        frame_count,
        frame_urls,
        csv_path
    )


# -----------------------------
# API endpoint: /detect (Video Upload)
# -----------------------------
@app.route('/detect', methods=['POST'])
def detect_video():
    if 'video' not in request.files:
        return jsonify({"error": "No video file uploaded."}), 400

    video = request.files['video']
    session_id = str(uuid.uuid4())

    video_path = os.path.join("uploads", f"{session_id}.mp4")
    video.save(video_path)

    try:
        video_url, frame_count, frame_urls, csv_path = process_video(video_path, session_id)
        response = {
            "message": "✅ Processing complete!",
            "processed_frames": frame_count,
            "output_video_url": video_url,
            "frames_urls": frame_urls,
            "keypoints_csv": f"http://127.0.0.1:5000/files/{os.path.basename(csv_path)}"
        }
        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -----------------------------
# Serve annotated videos
# -----------------------------
@app.route('/videos/<filename>')
def get_video(filename):
    return send_from_directory('outputs/videos', filename)


# -----------------------------
# Serve extracted frames
# -----------------------------
@app.route('/frames/<filename>')
def get_frame(filename):
    return send_from_directory('outputs/frames', filename)


# -----------------------------
# Serve keypoints CSV
# -----------------------------
@app.route('/files/<filename>')
def get_file(filename):
    return send_from_directory('outputs', filename)


@app.route('/')
def home():
    return jsonify({"message": "🎥 YOLOv8 Pose Detection Video API Running"})


# -----------------------------
# Run Flask App
# -----------------------------
if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
