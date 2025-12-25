"""
Bird Detection Module using YOLOv8, Roboflow, and Classical Computer Vision
Handles detection with multiple approaches for poultry scenarios
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Tuple
from inference_sdk import InferenceHTTPClient
import os


class BirdDetector:
    def __init__(self, model_path: str = 'models/yolov8n.pt', conf_thresh: float = 0.25, iou_thresh: float = 0.45):
        """
        Initialize bird detector
        
        Args:
            model_path: Path to YOLOv8 model weights
            conf_thresh: Confidence threshold for detections
            iou_thresh: IOU threshold for NMS
        """
        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        
        # Initialize Roboflow client
        try:
            self.roboflow_client = InferenceHTTPClient(
                api_url="https://serverless.roboflow.com",
                api_key="NRMWWjOJDAeSV7EKvqrQ"
            )
            self.use_roboflow = True
            print("✅ Roboflow client initialized")
        except Exception as e:
            self.roboflow_client = None
            self.use_roboflow = False
            print(f"⚠️  Roboflow unavailable: {e}")
    
    def detect_birds_roboflow(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect birds using Roboflow trained model (PRIMARY METHOD)
        
        Args:
            frame: Input BGR frame
            
        Returns:
            List of detections
        """
        if not self.use_roboflow:
            return []
        
        # Save frame temporarily
        temp_path = 'temp_detection.jpg'
        cv2.imwrite(temp_path, frame)
        
        try:
            # Run Roboflow inference
            result = self.roboflow_client.run_workflow(
                workspace_name="poultry-n9gkg",
                workflow_id="find-chickens-5",
                images={"image": temp_path},
                use_cache=False
            )
            
            detections = []
            
            if len(result) > 0 and 'predictions' in result[0]:
                predictions_data = result[0]['predictions']
                if 'predictions' in predictions_data:
                    for det in predictions_data['predictions']:
                        x = int(det['x'] - det['width']/2)
                        y = int(det['y'] - det['height']/2)
                        w = int(det['width'])
                        h = int(det['height'])
                        
                        bbox_center = (x + w // 2, y + h // 2)
                        depth_factor = self.get_depth_zone_factor(bbox_center, frame.shape)
                        
                        detections.append({
                            'bbox': [x, y, w, h],
                            'confidence': det['confidence'],
                            'depth_factor': depth_factor,
                            'center': bbox_center,
                            'class_id': 0
                        })
            
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            return detections
            
        except Exception as e:
            print(f"Roboflow detection error: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return []
    
    def estimate_weight(self, bbox: List[int]) -> float:
        """
        Estimate chicken weight based on bounding box area
        
        Args:
            bbox: [x, y, w, h]
            
        Returns:
            Estimated weight in kg
        """
        _, _, w, h = bbox
        area = w * h
        
        # Weight estimation based on area
        if area < 2000:
            weight = 0.8 + (area / 2000) * 0.4
        elif area < 4000:
            weight = 1.2 + ((area - 2000) / 2000) * 0.6
        else:
            weight = 1.8 + (min(area - 4000, 3000) / 3000) * 0.7
        
        return round(weight, 2)
        
    def detect_feeders(self, frame: np.ndarray) -> np.ndarray:
        """
        Detect red feeders/drinkers for occlusion masking
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Binary mask of feeder locations
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Red color range for feeders (two ranges for red hue wrap-around)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        feeder_mask = mask1 + mask2
        
        # Morphological operations to clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        feeder_mask = cv2.morphologyEx(feeder_mask, cv2.MORPH_CLOSE, kernel)
        
        return feeder_mask
    
    def get_depth_zone_factor(self, bbox_center: Tuple[int, int], frame_shape: Tuple[int, int]) -> float:
        """
        Calculate depth correction factor based on position in frame
        Accounts for perspective distortion (overhead camera)
        
        Args:
            bbox_center: (x, y) center of bounding box
            frame_shape: (height, width) of frame
            
        Returns:
            Depth correction factor
        """
        h, w = frame_shape[:2]
        center_x, center_y = w / 2, h / 2
        
        # Calculate distance from frame center
        dx = bbox_center[0] - center_x
        dy = bbox_center[1] - center_y
        distance = np.sqrt(dx**2 + dy**2)
        
        # Maximum distance (corner of frame)
        max_distance = np.sqrt((w/2)**2 + (h/2)**2)
        
        # Normalized distance (0 at center, 1 at corners)
        normalized_dist = distance / max_distance
        
        # Apply depth correction zones
        if normalized_dist < 0.3:  # Center zone
            return 1.0
        elif normalized_dist < 0.6:  # Middle zone
            return 1.05
        else:  # Outer zone
            return 1.12
    
    def detect_birds(self, frame: np.ndarray) -> List[Dict]:
        """
        Primary detection method - uses Roboflow if available, otherwise YOLO
        
        Args:
            frame: Input BGR frame
            
        Returns:
            List of detections
        """
        # Try Roboflow first
        if self.use_roboflow:
            detections = self.detect_birds_roboflow(frame)
            if len(detections) > 0:
                return detections
        
        # Fallback to YOLOv8
        return self.detect_birds_yolo(frame)
    
    def detect_birds_yolo(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect birds in frame using YOLOv8 (fallback method)
        
        Args:
            frame: Input BGR frame
            
        Returns:
            List of detections
        """
        results = self.model.predict(
            frame,
            conf=0.01,
            iou=self.iou_thresh,
            verbose=False,
            device='cpu',
            imgsz=640
        )
        
        detections = []
        
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                
                if confidence < 0.15:
                    continue
                
                x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                
                bbox_center = (x + w // 2, y + h // 2)
                depth_factor = self.get_depth_zone_factor(bbox_center, frame.shape)
                
                detections.append({
                    'bbox': [x, y, w, h],
                    'confidence': confidence,
                    'depth_factor': depth_factor,
                    'center': bbox_center,
                    'class_id': class_id
                })
        
        return detections
    
    def detect_birds_blob(self, frame: np.ndarray) -> List[Dict]:
        """
        Optimized blob detection for dense poultry scenarios
        
        Args:
            frame: Input BGR frame
            
        Returns:
            List of detections
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        
        dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        
        contours, _ = cv2.findContours(sure_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area < 500 or area > 20000:
                continue
            
            x, y, bbox_w, bbox_h = cv2.boundingRect(contour)
            
            aspect_ratio = bbox_w / max(bbox_h, 1)
            if aspect_ratio < 0.3 or aspect_ratio > 3.5:
                continue
            
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter ** 2)
            
            if circularity < 0.2:
                continue
            
            bbox_center = (x + bbox_w // 2, y + bbox_h // 2)
            depth_factor = self.get_depth_zone_factor(bbox_center, frame.shape)
            
            detections.append({
                'bbox': [x, y, bbox_w, bbox_h],
                'confidence': min(circularity * 1.2, 0.95),
                'depth_factor': depth_factor,
                'center': bbox_center,
                'class_id': 0
            })
        
        return detections
    
    def detect_birds_watershed(self, frame: np.ndarray) -> List[Dict]:
        """
        Use watershed algorithm to separate overlapping chickens
        
        Args:
            frame: Input BGR frame
            
        Returns:
            List of detections
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        
        sure_bg = cv2.dilate(opening, kernel, iterations=2)
        
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.2 * dist_transform.max(), 255, 0)
        
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        
        markers = cv2.watershed(frame, markers)
        
        detections = []
        unique_markers = np.unique(markers)
        
        for marker in unique_markers:
            if marker <= 1:
                continue
            
            mask = np.zeros_like(gray)
            mask[markers == marker] = 255
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) == 0:
                continue
            
            contour = contours[0]
            area = cv2.contourArea(contour)
            
            if area < 300 or area > 50000:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            
            aspect_ratio = w / max(h, 1)
            if aspect_ratio < 0.2 or aspect_ratio > 5.0:
                continue
            
            bbox_center = (x + w // 2, y + h // 2)
            depth_factor = self.get_depth_zone_factor(bbox_center, frame.shape)
            
            detections.append({
                'bbox': [x, y, w, h],
                'confidence': 0.85,
                'depth_factor': depth_factor,
                'center': bbox_center,
                'class_id': 0
            })
        
        return detections
    
    def detect_birds_contours(self, frame: np.ndarray) -> List[Dict]:
        """
        Simple contour-based detection
        
        Args:
            frame: Input BGR frame
            
        Returns:
            List of detections
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        
        binary = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                        cv2.THRESH_BINARY_INV, 31, 10)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel, iterations=1)
        
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area < 400 or area > 40000:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            
            aspect_ratio = w / max(h, 1)
            if aspect_ratio < 0.2 or aspect_ratio > 5.0:
                continue
            
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidity = area / hull_area
                if solidity < 0.5:
                    continue
            
            bbox_center = (x + w // 2, y + h // 2)
            depth_factor = self.get_depth_zone_factor(bbox_center, frame.shape)
            
            detections.append({
                'bbox': [x, y, w, h],
                'confidence': 0.80,
                'depth_factor': depth_factor,
                'center': bbox_center,
                'class_id': 0
            })
        
        return detections
    
    def detect_birds_hybrid(self, frame: np.ndarray) -> List[Dict]:
        """
        Hybrid detection combining multiple methods
        
        Args:
            frame: Input BGR frame
            
        Returns:
            List of detections with duplicates removed
        """
        all_detections = []
        
        # Try Roboflow first
        if self.use_roboflow:
            roboflow_dets = self.detect_birds_roboflow(frame)
            all_detections.extend(roboflow_dets)
        
        # Add watershed
        watershed_dets = self.detect_birds_watershed(frame)
        all_detections.extend(watershed_dets)
        
        # Add contours
        contour_dets = self.detect_birds_contours(frame)
        all_detections.extend(contour_dets)
        
        if len(all_detections) == 0:
            return []
        
        # NMS to remove duplicates
        boxes = np.array([det['bbox'] for det in all_detections])
        confidences = np.array([det['confidence'] for det in all_detections])
        
        boxes_xyxy = boxes.copy()
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]
        
        indices = cv2.dnn.NMSBoxes(
            boxes_xyxy.tolist(),
            confidences.tolist(),
            score_threshold=0.3,
            nms_threshold=0.4
        )
        
        final_detections = []
        if len(indices) > 0:
            indices = indices.flatten()
            for i in indices:
                final_detections.append(all_detections[i])
        
        return final_detections
    
    def detect_birds_motion_based(self, frame: np.ndarray, background_subtractor) -> List[Dict]:
        """
        Detect birds using background subtraction
        
        Args:
            frame: Input BGR frame
            background_subtractor: cv2.BackgroundSubtractor object
            
        Returns:
            List of detections
        """
        fg_mask = background_subtractor.apply(frame)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 800 or area > 15000:
                continue
            
            x, y, bbox_w, bbox_h = cv2.boundingRect(contour)
            
            aspect_ratio = bbox_w / max(bbox_h, 1)
            if aspect_ratio < 0.5 or aspect_ratio > 2.5:
                continue
            
            bbox_center = (x + bbox_w // 2, y + bbox_h // 2)
            depth_factor = self.get_depth_zone_factor(bbox_center, frame.shape)
            
            detections.append({
                'bbox': [x, y, bbox_w, bbox_h],
                'confidence': 0.8,
                'depth_factor': depth_factor,
                'center': bbox_center,
                'class_id': 0
            })
        
        return detections

    def filter_by_feeder_mask(self, detections: List[Dict], feeder_mask: np.ndarray, 
                               overlap_threshold: float = 0.5) -> List[Dict]:
        """
        Filter out detections occluded by feeders
        
        Args:
            detections: List of detection dictionaries
            feeder_mask: Binary mask of feeder locations
            overlap_threshold: Max allowed overlap ratio
            
        Returns:
            Filtered detection list with occlusion flags
        """
        filtered_detections = []
        
        for det in detections:
            x, y, w, h = det['bbox']
            
            if y + h > feeder_mask.shape[0] or x + w > feeder_mask.shape[1]:
                continue
            
            det_region = feeder_mask[y:y+h, x:x+w]
            
            if det_region.size > 0:
                overlap_ratio = np.sum(det_region > 0) / det_region.size
            else:
                overlap_ratio = 0
            
            det['occluded_by_feeder'] = overlap_ratio > overlap_threshold
            det['occlusion_ratio'] = overlap_ratio
            
            filtered_detections.append(det)
        
        return filtered_detections
