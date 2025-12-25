import cv2
import os

os.makedirs('dataset/train_images', exist_ok=True)

cap = cv2.VideoCapture('videos/test_video.mp4')
frame_count = 0
saved = 0

# Extract 50 diverse frames
while saved < 50:
    ret, frame = cap.read()
    if ret and frame_count % 20 == 0:  # Every 20th frame for diversity
        cv2.imwrite(f'dataset/train_images/frame_{saved:03d}.jpg', frame)
        saved += 1
        print(f"✓ Saved frame {saved}/50")
    frame_count += 1
    
    if not ret:
        break

cap.release()
print(f"\n✅ Extracted {saved} frames to dataset/train_images/")
print("📁 Open this folder and check the images")
