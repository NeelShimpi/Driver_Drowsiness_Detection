#!/usr/bin/env python3
"""
Flask Backend API for Drowsiness Detection Web App
Provides REST API endpoints for real-time drowsiness detection

Usage:
    python flask_backend.py

Then open web_app.html in your browser
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import torch
import numpy as np
import base64
import io
from PIL import Image
import time
import os
import sys

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Try to import full detector
try:
    sys.path.append(os.path.dirname(__file__))
    from drowsiness_detection import DrowsinessDetector, eye_aspect_ratio
    from scipy.spatial import distance as dist
    import dlib
    from imutils import face_utils
    USE_FULL_DETECTOR = True
    print("✓ Full detector system loaded")
except ImportError as e:
    USE_FULL_DETECTOR = False
    print(f"⚠ Full detector not available: {e}")
    print("  Using basic face detection only")


class BasicDetector:
    """Basic face detection fallback"""
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        print("✓ Basic detector initialized")
    
    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        return {
            'timestamp': time.time(),
            'faces_detected': len(faces),
            'faces': [{
                'bbox': {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)},
                'is_drowsy': False,
                'ear': None,
                'vit': None
            } for x, y, w, h in faces],
            'overall_drowsy': False
        }


class FullDetectorWrapper:
    """Wrapper for full drowsiness detection system"""
    def __init__(self):
        self.detector = DrowsinessDetector(use_cuda=torch.cuda.is_available())
        print(f"✓ Full detector initialized (CUDA: {torch.cuda.is_available()})")
    
    def process_frame(self, frame):
        """Process frame with full detection pipeline"""
        # EAR detection
        frame_annotated, ear_drowsy = self.detector.detect_drowsiness_ear(frame)
        
        # Get EAR value
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        ear_value = None
        if len(faces) > 0 and self.detector.predictor is not None:
            x, y, w, h = faces[0]
            rect = dlib.rectangle(int(x), int(y), int(x+w), int(y+h))
            try:
                shape = self.detector.predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)
                
                left_eye = shape[36:42]
                right_eye = shape[42:48]
                
                left_ear = eye_aspect_ratio(left_eye)
                right_ear = eye_aspect_ratio(right_eye)
                ear_value = (left_ear + right_ear) / 2.0
            except:
                pass
        
        # ViT detection
        vit_prob, vit_class = self.detector.detect_drowsiness_vit(frame)
        vit_drowsy = (vit_class in [1, 2, 5]) and (vit_prob > 0.70)
        
        # Combined result
        is_drowsy = ear_drowsy or vit_drowsy
        
        return {
            'timestamp': time.time(),
            'faces_detected': len(faces),
            'faces': [{
                'bbox': {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)},
                'is_drowsy': is_drowsy,
                'ear': {'ear': float(ear_value), 'is_drowsy': ear_drowsy} if ear_value else None,
                'vit': {
                    'drowsy_probability': float(vit_prob),
                    'class': int(vit_class),
                    'is_drowsy': vit_drowsy
                }
            } for x, y, w, h in faces],
            'overall_drowsy': is_drowsy
        }


# Initialize detector
if USE_FULL_DETECTOR:
    try:
        detector = FullDetectorWrapper()
    except Exception as e:
        print(f"⚠ Failed to init full detector: {e}")
        print("  Falling back to basic detection")
        detector = BasicDetector()
        USE_FULL_DETECTOR = False
else:
    detector = BasicDetector()


# Sessions storage (in production, use Redis or database)
sessions = {}


# ==================== API ROUTES ====================

@app.route('/')
def index():
    """Serve the web interface"""
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), 'web_app.html')


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'detector_type': 'full' if USE_FULL_DETECTOR else 'basic',
        'cuda_available': torch.cuda.is_available() if USE_FULL_DETECTOR else False,
        'timestamp': time.time()
    })


@app.route('/api/detect', methods=['POST'])
def detect():
    """
    Main detection endpoint
    Accepts base64 encoded image and returns detection results
    """
    try:
        # Get image from request
        if 'image' not in request.json:
            return jsonify({'error': 'No image provided'}), 400
        
        # Decode base64 image
        image_data = request.json['image']
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Process frame
        start_time = time.time()
        results = detector.process_frame(frame)
        processing_time = (time.time() - start_time) * 1000
        
        results['processing_time_ms'] = processing_time
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/session/start', methods=['POST'])
def start_session():
    """Start a new detection session"""
    session_id = str(int(time.time() * 1000))
    
    sessions[session_id] = {
        'start_time': time.time(),
        'total_frames': 0,
        'drowsy_frames': 0,
        'alert_frames': 0
    }
    
    return jsonify({
        'session_id': session_id,
        'message': 'Session started successfully'
    })


@app.route('/api/session/<session_id>/update', methods=['POST'])
def update_session(session_id):
    """Update session with new detection result"""
    if session_id not in sessions:
        return jsonify({'error': 'Session not found'}), 404
    
    try:
        data = request.json
        session = sessions[session_id]
        
        session['total_frames'] += 1
        
        if data.get('is_drowsy'):
            session['drowsy_frames'] += 1
        else:
            session['alert_frames'] += 1
        
        return jsonify({'success': True})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/session/<session_id>/stats', methods=['GET'])
def get_session_stats(session_id):
    """Get session statistics"""
    if session_id not in sessions:
        return jsonify({'error': 'Session not found'}), 404
    
    session = sessions[session_id]
    total = session['total_frames']
    
    stats = {
        'session_id': session_id,
        'duration_seconds': time.time() - session['start_time'],
        'total_frames': total,
        'drowsy_frames': session['drowsy_frames'],
        'alert_frames': session['alert_frames'],
        'drowsy_percentage': (session['drowsy_frames'] / total * 100) if total > 0 else 0,
        'alert_percentage': (session['alert_frames'] / total * 100) if total > 0 else 0
    }
    
    return jsonify(stats)


@app.route('/api/config', methods=['GET'])
def get_config():
    """Get current detector configuration"""
    if USE_FULL_DETECTOR:
        return jsonify({
            'detector_type': 'full',
            'ear_threshold': detector.detector.EAR_THRESHOLD,
            'consecutive_frames': detector.detector.EAR_CONSEC_FRAMES,
            'device': str(detector.detector.device)
        })
    else:
        return jsonify({
            'detector_type': 'basic',
            'message': 'Basic detector - no configuration available'
        })


# ==================== MAIN ====================

if __name__ == '__main__':
    # Print startup info
    print("\n" + "="*70)
    print(" " * 20 + "🚀 DROWSINESS DETECTION API")
    print("="*70)
    print(f"\nDetector Type: {'Full System' if USE_FULL_DETECTOR else 'Basic Face Detection'}")
    if USE_FULL_DETECTOR:
        print(f"CUDA Available: {torch.cuda.is_available()}")
    
    print("\nServer Configuration:")
    print(f"  Host: 0.0.0.0")
    print(f"  Port: 5000")
    print(f"  URL:  http://localhost:5000")
    
    print("\nAPI Endpoints:")
    print(f"  GET  /api/health")
    print(f"  POST /api/detect")
    print(f"  POST /api/session/start")
    print(f"  GET  /api/config")
    
    print("\n📱 Access from other devices:")
    print(f"  1. Find your IP: ifconfig (Mac/Linux) or ipconfig (Windows)")
    print(f"  2. Open: http://YOUR_IP:5000 on other devices")
    
    print("\n💡 Next Steps:")
    print(f"  1. Open web_app.html in your browser")
    print(f"  2. Grant camera permissions")
    print(f"  3. Click 'Start Detection'")
    
    print("\n" + "="*70 + "\n")
    
    # Check dependencies
    try:
        from flask_cors import CORS
    except ImportError:
        print("⚠ Installing flask-cors...")
        import subprocess
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'flask-cors'])
        from flask_cors import CORS
        CORS(app)
    
    # Run server
    try:
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=False,
            threaded=True
        )
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server error: {e}")
