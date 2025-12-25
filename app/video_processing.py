"""
Video Processing Module
Handles video annotation and output generation
"""

import cv2
import numpy as np
from typing import List, Dict
import os

class VideoProcessor:
    def __init__(self, output_dir: str = 'outputs'):
        """Initialize video processor"""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def annotate_frame(self, frame: np.ndarray, tracked_detections: List[Dict], 
                       count: int, timestamp: str) -> np.ndarray:
        """
        Annotate frame with bounding boxes, IDs, and count
        
        Args:
            frame: Input frame
            tracked_detections: List of tracked detections
            count: Current bird count
            timestamp: Current timestamp string
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        # Draw detections
        for det in tracked_detections:
            x, y, w, h = det['bbox']
            track_id = det.get('track_id', -1)
            confidence = det['confidence']
            
            # Draw bounding box
            color = (0, 255, 0)  # Green
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            
            # Draw track ID and confidence
            label = f"ID:{track_id} {confidence:.2f}"
            cv2.putText(annotated, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw count overlay
        overlay_height = 80
        overlay = annotated.copy()
        cv2.rectangle(overlay, (10, 10), (400, overlay_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated)
        
        # Add text
        cv2.putText(annotated, f"Count: {count}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        cv2.putText(annotated, f"Time: {timestamp}", (20, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return annotated
    
    def process_video(self, video_path: str, detector, tracker, weight_estimator,
                     fps_sample: int = 5, max_frames: int = None, detection_method: str = "blob") -> Dict:
        """
        Process entire video with detection, tracking, and weight estimation
        
        Args:
            video_path: Path to input video
            detector: BirdDetector instance
            tracker: BirdTracker instance
            weight_estimator: WeightEstimator instance
            fps_sample: Sample every N frames
            max_frames: Maximum frames to process (None = all)
            detection_method: "yolo" or "blob" (default: "blob")
            
        Returns:
            Dictionary with results
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return {'error': 'Failed to open video'}
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Output video writer
        output_path = os.path.join(self.output_dir, 'annotated_output.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps // fps_sample, (width, height))
        
        # Results storage
        counts_timeseries = []
        tracks_sample = []
        frame_id = 0
        processed_frames = 0
        
        print(f"Processing video: {total_frames} frames at {fps} FPS")
        print(f"Sampling every {fps_sample} frames...")
        print(f"Using detection method: {detection_method}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample frames
            if frame_id % fps_sample != 0:
                frame_id += 1
                continue
            
            # Calculate timestamp
            timestamp_sec = frame_id / fps
            timestamp = f"{int(timestamp_sec // 60):02d}:{int(timestamp_sec % 60):02d}"
                # Detect birds using specified method
            if detection_method == "hybrid":
                detections = detector.detect_birds_hybrid(frame)
            elif detection_method == "watershed":
                detections = detector.detect_birds_watershed(frame)
            elif detection_method == "blob":
                detections = detector.detect_birds_blob(frame)
            else:
                detections = detector.detect_birds(frame)
            # Detect birds using specified method
            # Track birds
            tracked_detections = tracker.update(detections, frame_id)
            
            # Count stable tracks
            stable_tracks = tracker.get_stable_tracks(min_frames=5)
            count = len(stable_tracks)
            
            # Store count
            counts_timeseries.append({
                'timestamp': timestamp,
                'count': count
            })
            
            # Store sample tracks (first 10 frames)
            if processed_frames < 10:
                for det in tracked_detections[:5]:  # Sample first 5
                    tracks_sample.append({
                        'track_id': det.get('track_id', -1),
                        'bbox': det['bbox'],
                        'confidence': round(det['confidence'], 3)
                    })
            
            # Annotate frame
            annotated = self.annotate_frame(frame, tracked_detections, count, timestamp)
            out.write(annotated)
            
            processed_frames += 1
            
            if processed_frames % 30 == 0:
                print(f"Processed {processed_frames} frames, Current count: {count}")
            
            # Break if max frames reached
            if max_frames and processed_frames >= max_frames:
                break
            
            frame_id += 1
        
        cap.release()
        out.release()
        
        print(f"\n✓ Processed {processed_frames} frames")
        print(f"✓ Video saved to: {output_path}")
        
        # Weight estimation on last frame
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(frame_id - 1, total_frames - 1))
        ret, last_frame = cap.read()
        cap.release()
        
        if ret:
            # Use same detection method for weight estimation
            if detection_method == "blob":
                last_detections = detector.detect_birds_blob(last_frame)
            else:
                last_detections = detector.detect_birds(last_frame)
            
            last_tracked = tracker.update(last_detections, frame_id)
            weight_estimates = weight_estimator.estimate_weights(
                last_frame, last_tracked, tracker.track_history
            )
        else:
            weight_estimates = {'per_bird': [], 'aggregate_avg': 0.0, 'unit': 'index (0-100)'}
        
        # Save counts to CSV
        import csv
        csv_path = os.path.join(self.output_dir, 'counts_timeseries.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['timestamp', 'count'])
            writer.writeheader()
            writer.writerows(counts_timeseries)
        
        print(f"✓ Counts saved to: {csv_path}")
        
        return {
            'counts': counts_timeseries,
            'tracks_sample': tracks_sample[:20],  # Limit to 20
            'weight_estimates': weight_estimates,
            'artifacts': {
                'annotated_video': output_path,
                'counts_csv': csv_path
            }
        }
