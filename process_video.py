"""
Process entire video with Roboflow detection
"""

from app.detection import BirdDetector
import cv2
import os

def main():
    print("\n" + "="*60)
    print("🎬 PROCESSING FULL VIDEO")
    print("="*60 + "\n")
    
    detector = BirdDetector()
    video_path = 'videos/test_video.mp4'
    output_path = 'outputs/final_video.mp4'
    
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    frame_skip = 10  # Process every 10th frame
    
    all_counts = []
    all_weights = []
    
    print(f"📹 Video: {video_path}")
    print(f"📊 Total frames: {total_frames}")
    print(f"⚡ Processing every {frame_skip}th frame\n")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_skip == 0:
            # Detect
            detections = detector.detect_birds(frame)
            count = len(detections)
            all_counts.append(count)
            
            # Draw
            frame_weights = []
            for det in detections:
                x, y, w, h = det['bbox']
                weight = detector.estimate_weight(det['bbox'])
                frame_weights.append(weight)
                
                color = (0, 255, 0) if det['confidence'] > 0.6 else (0, 255, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, f"{weight}kg", (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            all_weights.extend(frame_weights)
            avg_weight = sum(frame_weights) / len(frame_weights) if frame_weights else 0
            
            # Summary text
            cv2.putText(frame, f"Count: {count}", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            cv2.putText(frame, f"Avg: {avg_weight:.2f}kg", (50, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            print(f"✓ Frame {frame_count}/{total_frames}: {count} chickens, {avg_weight:.2f}kg avg")
        
        out.write(frame)
        frame_count += 1
    
    cap.release()
    out.release()
    
    # Final stats
    avg_count = sum(all_counts) / len(all_counts) if all_counts else 0
    max_count = max(all_counts) if all_counts else 0
    total_weight = sum(all_weights)
    avg_weight_per_bird = sum(all_weights) / len(all_weights) if all_weights else 0
    
    print("\n" + "="*60)
    print("📋 FINAL REPORT")
    print("="*60)
    print(f"✅ Frames processed: {len(all_counts)}")
    print(f"🐔 Average count: {avg_count:.1f} chickens")
    print(f"🐔 Max count: {max_count} chickens")
    print(f"⚖️  Total weight: {total_weight:.2f} kg")
    print(f"⚖️  Avg weight/bird: {avg_weight_per_bird:.2f} kg")
    print(f"💾 Video saved: {output_path}")
    print("="*60 + "\n")
    
    # Save report
    with open('outputs/video_report.txt', 'w') as f:
        f.write("CHICKEN DETECTION & WEIGHT ESTIMATION REPORT\n")
        f.write("="*60 + "\n\n")
        f.write(f"Video: {video_path}\n")
        f.write(f"Total Frames: {total_frames}\n")
        f.write(f"Processed Frames: {len(all_counts)}\n\n")
        f.write(f"Average Count: {avg_count:.1f} chickens\n")
        f.write(f"Maximum Count: {max_count} chickens\n")
        f.write(f"Total Weight: {total_weight:.2f} kg\n")
        f.write(f"Avg Weight/Bird: {avg_weight_per_bird:.2f} kg\n")
    
    print("📄 Report saved: outputs/video_report.txt\n")

if __name__ == "__main__":
    main()
