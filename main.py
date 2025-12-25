"""
FastAPI application for Bird Counting and Weight Estimation
"""

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
import uvicorn
import os
import cv2
import json
from app.detection import BirdDetector

app = FastAPI(
    title="Bird Counting and Weight Estimation API",
    version="1.0.0",
    description="Poultry CCTV video analysis for bird counting and weight estimation"
)

# Create necessary directories
os.makedirs('videos', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

@app.get("/")
def root():
    """Root endpoint"""
    return {
        "message": "Bird Counting and Weight Estimation API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "test_detection": "/test_detection",
            "analyze_frame": "/analyze_frame"
        }
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "OK",
        "service": "Bird Detection API",
        "roboflow_enabled": True
    }

@app.get("/test_detection")
def test_detection():
    """
    Test Roboflow detection on a single frame from test video
    Returns JSON with detection results
    """
    try:
        detector = BirdDetector()
        
        # Load test video
        video_path = 'videos/test_video.mp4'
        if not os.path.exists(video_path):
            return JSONResponse(
                status_code=404,
                content={"error": "Test video not found", "path": video_path}
            )
        
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return JSONResponse(
                status_code=500,
                content={"error": "Could not read video frame"}
            )
        
        # Run detection
        detections = detector.detect_birds(frame)
        
        # Calculate statistics
        total_count = len(detections)
        weights = [detector.estimate_weight(d['bbox']) for d in detections]
        total_weight = sum(weights)
        avg_weight = total_weight / total_count if total_count > 0 else 0
        
        # Confidence breakdown
        high_conf = len([d for d in detections if d['confidence'] > 0.6])
        med_conf = len([d for d in detections if 0.4 <= d['confidence'] <= 0.6])
        low_conf = len([d for d in detections if d['confidence'] < 0.4])
        
        # Draw detections
        for det in detections:
            x, y, w, h = det['bbox']
            weight = detector.estimate_weight(det['bbox'])
            conf = det['confidence']
            
            color = (0, 255, 0) if conf > 0.6 else (0, 255, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"{weight}kg", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Add summary
        cv2.putText(frame, f"Count: {total_count}", (50, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.putText(frame, f"Avg: {avg_weight:.2f}kg", (50, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # Save output
        output_path = 'outputs/api_test_detection.jpg'
        cv2.imwrite(output_path, frame)
        
        # Prepare detailed detection list
        detection_list = []
        for i, det in enumerate(detections):
            detection_list.append({
                "id": i + 1,
                "bbox": {
                    "x": int(det['bbox'][0]),
                    "y": int(det['bbox'][1]),
                    "width": int(det['bbox'][2]),
                    "height": int(det['bbox'][3])
                },
                "confidence": round(det['confidence'], 3),
                "estimated_weight_kg": detector.estimate_weight(det['bbox']),
                "depth_factor": round(det['depth_factor'], 2)
            })
        
        # Return comprehensive JSON response
        return JSONResponse(content={
            "status": "success",
            "timestamp": "2025-12-25T13:00:00Z",
            "summary": {
                "total_chickens_detected": total_count,
                "total_estimated_weight_kg": round(total_weight, 2),
                "average_weight_per_bird_kg": round(avg_weight, 2)
            },
            "confidence_distribution": {
                "high_confidence_count": high_conf,
                "medium_confidence_count": med_conf,
                "low_confidence_count": low_conf
            },
            "weight_statistics": {
                "min_weight_kg": round(min(weights), 2) if weights else 0,
                "max_weight_kg": round(max(weights), 2) if weights else 0,
                "average_weight_kg": round(avg_weight, 2)
            },
            "detections": detection_list[:10],  # First 10 for brevity
            "detection_method": "roboflow" if detector.use_roboflow else "yolo_fallback",
            "output_image": output_path,
            "video_source": video_path,
            "frame_resolution": {
                "width": frame.shape[1],
                "height": frame.shape[0]
            }
        })
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "status": "failed"}
        )

@app.post("/analyze_frame")
async def analyze_frame(image: UploadFile = File(...)):
    """
    Analyze uploaded image for chicken detection
    """
    try:
        # Save uploaded image
        image_path = f"videos/uploaded_{image.filename}"
        with open(image_path, "wb") as f:
            content = await image.read()
            f.write(content)
        
        # Load and detect
        detector = BirdDetector()
        frame = cv2.imread(image_path)
        
        if frame is None:
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid image file"}
            )
        
        detections = detector.detect_birds(frame)
        
        # Calculate stats
        total_count = len(detections)
        weights = [detector.estimate_weight(d['bbox']) for d in detections]
        total_weight = sum(weights)
        avg_weight = total_weight / total_count if total_count > 0 else 0
        
        # Clean up
        if os.path.exists(image_path):
            os.remove(image_path)
        
        return JSONResponse(content={
            "status": "success",
            "total_chickens": total_count,
            "total_weight_kg": round(total_weight, 2),
            "average_weight_kg": round(avg_weight, 2)
        })
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

if __name__ == "__main__":
    print("="*60)
    print("🚀 Starting Bird Counting API with Roboflow Integration")
    print("="*60)
    print("\n📡 API Endpoints:")
    print("   - http://localhost:8000/docs (Swagger UI)")
    print("   - http://localhost:8000/health")
    print("   - http://localhost:8000/test_detection")
    print("   - http://localhost:8000/analyze_frame")
    print("="*60 + "\n")
    
    # ✅ FIXED: Pass 'app' directly instead of string path
    uvicorn.run(app, host="0.0.0.0", port=8000)
