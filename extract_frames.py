import cv2, os, glob

# Folder containing your videos
videos_path = "Yoga_vid_Collected"
# Output folder for frames
output_folder = "frames_dataset"

# Settings to save space
resize_width = 640   # reduce width to 640px (original might be 1920)
resize_height = 360  # reduce height to 360px
frame_skip = 10      # save only every 10th frame (adjust as needed)

os.makedirs(output_folder, exist_ok=True)

for video_file in glob.glob(f"{videos_path}/*.mp4"):
    video_name = os.path.splitext(os.path.basename(video_file))[0]
    cap = cv2.VideoCapture(video_file)
    frame_id = 0
    saved = 0
    video_out = f"{output_folder}/{video_name}"
    os.makedirs(video_out, exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames to reduce total count
        if frame_id % frame_skip == 0:
            # Resize to smaller resolution
            resized = cv2.resize(frame, (resize_width, resize_height))
            cv2.imwrite(f"{video_out}/frame_{saved:04d}.jpg", resized)
            saved += 1

        frame_id += 1

    cap.release()
    print(f"✅ {video_name}: saved {saved} frames (reduced size {resize_width}x{resize_height})")

print("🎉 All videos processed successfully!")
