"""
Bird Tracking Module using Supervision ByteTrack
"""

import numpy as np
from typing import List, Dict
import supervision as sv

class BirdTracker:
    def __init__(self, track_thresh: float = 0.25, track_buffer: int = 30, match_thresh: float = 0.8):
        """
        Initialize ByteTrack tracker
        
        Args:
            track_thresh: Detection confidence threshold
            track_buffer: Number of frames to keep lost tracks
            match_thresh: IOU threshold for matching
        """
        self.tracker = sv.ByteTrack(
            track_activation_threshold=track_thresh,
            lost_track_buffer=track_buffer,
            minimum_matching_threshold=match_thresh,
            frame_rate=5
        )
        self.track_history = {}
        
    def update(self, detections: List[Dict], frame_id: int) -> List[Dict]:
        """
        Update tracker with new detections
        
        Args:
            detections: List of detection dictionaries
            frame_id: Current frame number
            
        Returns:
            List of tracked detections with IDs
        """
        if len(detections) == 0:
            return []
        
        # Convert to supervision format
        xyxy = []
        confidences = []
        
        for det in detections:
            x, y, w, h = det['bbox']
            xyxy.append([x, y, x + w, y + h])
            confidences.append(det['confidence'])
        
        xyxy = np.array(xyxy)
        confidences = np.array(confidences)
        
        # Create supervision Detections object
        sv_detections = sv.Detections(
            xyxy=xyxy,
            confidence=confidences
        )
        
        # Update tracker
        sv_detections = self.tracker.update_with_detections(sv_detections)
        
        # Convert back to our format with track IDs
        tracked_detections = []
        for i, track_id in enumerate(sv_detections.tracker_id):
            det = detections[i].copy()
            det['track_id'] = int(track_id)
            
            # Update track history
            if track_id not in self.track_history:
                self.track_history[track_id] = []
            
            self.track_history[track_id].append({
                'frame_id': frame_id,
                'bbox': det['bbox'],
                'confidence': det['confidence']
            })
            
            tracked_detections.append(det)
        
        return tracked_detections
    
    def get_stable_tracks(self, min_frames: int = 10) -> List[int]:
        """
        Get track IDs that have been stable for minimum frames
        
        Args:
            min_frames: Minimum number of frames for stable track
            
        Returns:
            List of stable track IDs
        """
        stable_ids = []
        for track_id, history in self.track_history.items():
            if len(history) >= min_frames:
                stable_ids.append(track_id)
        return stable_ids

