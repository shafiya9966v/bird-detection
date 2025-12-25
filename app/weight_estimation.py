"""
Weight Estimation Module
Estimates bird weight using size, depth, and appearance features
"""

import numpy as np
from typing import List, Dict
import cv2


class WeightEstimator:
    def __init__(self):
        """Initialize weight estimator"""
        # Average chicken weight range (grams) for calibration
        self.min_weight_g = 800    # Small chicken
        self.max_weight_g = 2500   # Large chicken
        
    def calculate_corrected_area(self, bbox: List[int], depth_factor: float) -> float:
        """
        Calculate depth-corrected bounding box area
        """
        x, y, w, h = bbox
        raw_area = w * h
        corrected_area = raw_area * depth_factor
        return corrected_area
    
    def extract_appearance_features(self, frame: np.ndarray, bbox: List[int]) -> Dict:
        """
        Extract appearance-based features for weight estimation
        """
        x, y, w, h = bbox
        
        # Ensure bbox is within frame
        h_frame, w_frame = frame.shape[:2]
        x = max(0, min(x, w_frame - 1))
        y = max(0, min(y, h_frame - 1))
        w = max(1, min(w, w_frame - x))
        h = max(1, min(h, h_frame - y))
        
        roi = frame[y:y+h, x:x+w]
        
        if roi.size == 0:
            return {
                'mean_intensity': 128,
                'std_intensity': 30,
                'color_density': 0.5
            }
        
        # Convert to grayscale for intensity
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Color intensity features
        mean_intensity = np.mean(gray_roi)
        std_intensity = np.std(gray_roi)
        
        # Color density (how "filled" the bbox is)
        _, binary = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        color_density = np.sum(binary > 0) / binary.size
        
        return {
            'mean_intensity': mean_intensity,
            'std_intensity': std_intensity,
            'color_density': color_density
        }
    
    def estimate_weight_for_bird(self, frame: np.ndarray, detection: Dict, 
                                 track_history: Dict) -> Dict:
        """
        Estimate weight for a single bird
        """
        bbox = detection['bbox']
        depth_factor = detection.get('depth_factor', 1.0)
        track_id = detection.get('track_id', -1)
        
        # Calculate corrected area
        corrected_area = self.calculate_corrected_area(bbox, depth_factor)
        
        # Get appearance features
        appearance = self.extract_appearance_features(frame, bbox)
        
        # Temporal averaging if track history exists
        temporal_avg_area = corrected_area
        if track_id in track_history and len(track_history[track_id]) > 0:
            history_areas = []
            for hist_det in track_history[track_id][-10:]:  # Last 10 frames
                hist_area = self.calculate_corrected_area(
                    hist_det['bbox'], 
                    hist_det.get('depth_factor', 1.0)
                )
                history_areas.append(hist_area)
            temporal_avg_area = np.mean(history_areas) if history_areas else corrected_area
        
        # Weight estimation formula
        # Normalize features
        norm_area = min(corrected_area / 40000.0, 1.0)  # Normalize to 0-1
        norm_temporal = min(temporal_avg_area / 40000.0, 1.0)
        norm_intensity = appearance['mean_intensity'] / 255.0
        norm_density = appearance['color_density']
        
        # Weighted combination
        weight_index = (
            0.45 * norm_area +           # Current size
            0.30 * norm_temporal +       # Temporal average
            0.15 * norm_intensity +      # Brightness (feather density)
            0.10 * norm_density          # Fill ratio
        ) * 100  # Scale to 0-100
        
        # Clamp to valid range
        weight_index = max(0, min(100, weight_index))
        
        # Convert to grams (linear mapping)
        weight_grams = self.min_weight_g + (weight_index / 100.0) * (self.max_weight_g - self.min_weight_g)
        
        # Confidence based on temporal stability
        confidence = 0.5
        if track_id in track_history and len(track_history[track_id]) > 5:
            # Higher confidence for well-tracked birds
            confidence = min(0.9, 0.5 + len(track_history[track_id]) * 0.02)
        
        return {
            'track_id': track_id,
            'weight_index': round(weight_index, 2),
            'weight_grams': round(weight_grams, 0),
            'confidence': round(confidence, 4),
            'features': {
                'area': corrected_area,
                'temporal_avg': temporal_avg_area,
                'intensity': appearance['mean_intensity']
            }
        }
    
    def estimate_weights(self, frame: np.ndarray, detections: List[Dict], 
                        track_history: Dict) -> Dict:
        """
        Estimate weights for all detected birds
        """
        per_bird_estimates = []
        
        for det in detections:
            estimate = self.estimate_weight_for_bird(frame, det, track_history)
            per_bird_estimates.append(estimate)
        
        # Calculate aggregate statistics
        if per_bird_estimates:
            weights_grams = [est['weight_grams'] for est in per_bird_estimates]
            weights_index = [est['weight_index'] for est in per_bird_estimates]
            
            aggregate_stats = {
                'avg_weight_grams': round(np.mean(weights_grams), 0),
                'avg_weight_index': round(np.mean(weights_index), 2),
                'min_weight_grams': round(np.min(weights_grams), 0),
                'max_weight_grams': round(np.max(weights_grams), 0),
                'std_weight_grams': round(np.std(weights_grams), 0)
            }
        else:
            aggregate_stats = {
                'avg_weight_grams': 0,
                'avg_weight_index': 0,
                'min_weight_grams': 0,
                'max_weight_grams': 0,
                'std_weight_grams': 0
            }
        
        return {
            'per_bird': per_bird_estimates,
            'aggregate': aggregate_stats,
            'unit': 'grams (estimated)',
            'calibration_note': 'Weights are estimated. Calibrate with 10-15 manual weighings for accuracy.'
        }
