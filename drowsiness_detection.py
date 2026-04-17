import cv2
import torch
import torch.nn as nn
import numpy as np
from scipy.spatial import distance as dist
from imutils import face_utils
import dlib
import time
import warnings
warnings.filterwarnings('ignore')

# --- HARDWARE COMPATIBILITY FIX FOR RTX 50-SERIES ---
# Disable pre-compiled fused attention kernels that crash on sm_120
if torch.cuda.is_available():
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
# ----------------------------------------------------

# Eye Aspect Ratio calculation
def eye_aspect_ratio(eye):
    """
    Calculate the Eye Aspect Ratio (EAR)
    EAR = (||p2-p6|| + ||p3-p5||) / (2||p1-p4||)
    """
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


# Mouth Aspect Ratio calculation
def mouth_aspect_ratio(mouth):
    """
    Calculate the Mouth Aspect Ratio (MAR)
    Uses the inner lip landmarks (points 60-67 in dlib's 68 landmark model)
    MAR = (||p2-p8|| + ||p3-p7|| + ||p4-p6||) / (2 * ||p1-p5||)
    
    A high MAR value indicates an open mouth (yawn).
    """
    # Vertical distances (inner lip top to bottom)
    A = dist.euclidean(mouth[13], mouth[19])  # 61 -> 67
    B = dist.euclidean(mouth[14], mouth[18])  # 62 -> 66
    C = dist.euclidean(mouth[15], mouth[17])  # 63 -> 65
    # Horizontal distance
    D = dist.euclidean(mouth[12], mouth[16])  # 60 -> 64
    mar = (A + B + C) / (2.0 * D)
    return mar


class DrowsinessDetector:
    def __init__(self, 
                 haar_cascade_path='haarcascade_frontalface_default.xml',
                 shape_predictor_path='shape_predictor_68_face_landmarks.dat',
                 vit_model_path='drowsiness_model_timm.pth',
                 use_cuda=True):
        """
        Initialize the drowsiness detection system
        
        Args:
            haar_cascade_path: Path to Haar Cascade XML file
            shape_predictor_path: Path to dlib's facial landmark predictor
            vit_model_path: Path to trained Vision Transformer model
            use_cuda: Whether to use CUDA for GPU acceleration
        """
        # CUDA setup
        self.device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Haar Cascade for face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + haar_cascade_path)
        
        # Dlib for facial landmarks
        self.detector = dlib.get_frontal_face_detector()
        try:
            self.predictor = dlib.shape_predictor(shape_predictor_path)
        except:
            print("Warning: dlib shape predictor not found. Download from:")
            print("http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
            self.predictor = None
        
        # Eye indices for 68-point facial landmarks
        self.LEFT_EYE_START, self.LEFT_EYE_END = 42, 48
        self.RIGHT_EYE_START, self.RIGHT_EYE_END = 36, 42
        
        # Mouth indices for 68-point facial landmarks (outer + inner lips)
        self.MOUTH_START, self.MOUTH_END = 48, 68
        
        # EAR thresholds
        self.EAR_THRESHOLD = 0.25
        self.EAR_CONSEC_FRAMES = 20
        self.frame_counter = 0
        self.drowsy_flag = False
        self.last_alarm_time = 0
        
        # MAR (yawn) thresholds
        self.MAR_THRESHOLD = 0.55       # Lowered to catch yawns more easily
        self.YAWN_CONSEC_FRAMES = 5     # Reduced from 15 to trigger faster
        self.yawn_frame_counter = 0
        self.yawn_flag = False
        self.total_yawns = 0
        
        import timm
        # Creating true Vision Transformer as requested
        self.vit_model = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=6)
        self.vit_model = self.vit_model.to(self.device)
        
        # Load pretrained weights if available
        try:
            self.vit_model.load_state_dict(torch.load(vit_model_path, map_location=self.device))
            self.vit_model.eval()
            print("Vision Transformer model loaded successfully")
        except Exception as e:
            print(f"Warning: Failed to load ViT model from {vit_model_path}")
            print(f"Error details: {e}")
            print("Using random initialization. For production, train the model first using train_vit.py")
        
        # Statistics
        self.total_frames = 0
        self.drowsy_frames = 0
        self.alert_frames = 0
        self.yawn_frames = 0
        self.true_positives = 0
        self.true_negatives = 0
        self.false_positives = 0
        self.false_negatives = 0
        
        # Performance metrics
        self.ear_history = []
        self.mar_history = []
        self.vit_confidence_history = []
        self.detection_accuracy = 0.0
        
    
    def detect_drowsiness_ear(self, frame):
        """
        Detect drowsiness using Eye Aspect Ratio method
        
        Args:
            frame: Input video frame
            
        Returns:
            frame: Annotated frame
            is_drowsy: Boolean indicating drowsiness
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces using Haar Cascade
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        is_drowsy = False
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            if self.predictor is None:
                continue
            
            # Get facial landmarks
            rect = dlib.rectangle(int(x), int(y), int(x+w), int(y+h))
            shape = self.predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            
            # Extract eye coordinates
            left_eye = shape[self.LEFT_EYE_START:self.LEFT_EYE_END]
            right_eye = shape[self.RIGHT_EYE_START:self.RIGHT_EYE_END]
            
            # Calculate EAR for both eyes
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0
            
            # Draw eye contours
            left_eye_hull = cv2.convexHull(left_eye)
            right_eye_hull = cv2.convexHull(right_eye)
            cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)
            
            # Check for drowsiness
            if ear < self.EAR_THRESHOLD:
                self.frame_counter += 1
                
                if self.frame_counter >= self.EAR_CONSEC_FRAMES:
                    is_drowsy = True
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                self.frame_counter = 0
            
            # Display EAR value
            cv2.putText(frame, f"EAR: {ear:.2f}", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return frame, is_drowsy
    
    
    def detect_yawn(self, frame):
        """
        Detect yawning using Mouth Aspect Ratio (MAR)
        
        The MAR is computed from the inner lip landmarks of dlib's
        68-point facial landmark model. When the mouth opens wide
        (as in a yawn), the MAR exceeds MOUTH_AR_THRESHOLD.
        
        Args:
            frame: Input video frame
            
        Returns:
            frame: Annotated frame with mouth contour and yawn status
            is_yawning: Boolean indicating whether a yawn is detected
            mar_value: The current Mouth Aspect Ratio value
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        is_yawning = False
        mar_value = 0.0
        
        for (x, y, w, h) in faces:
            if self.predictor is None:
                continue
            
            # Get facial landmarks
            rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
            shape = self.predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            
            # Extract mouth coordinates (points 48-67)
            mouth = shape[self.MOUTH_START:self.MOUTH_END]
            
            # Calculate MAR
            mar_value = mouth_aspect_ratio(mouth)
            
            # Draw mouth contour
            mouth_hull = cv2.convexHull(mouth)
            cv2.drawContours(frame, [mouth_hull], -1, (255, 0, 255), 1)
            
            # Check for yawn
            if mar_value > self.MAR_THRESHOLD:
                self.yawn_frame_counter += 1
                
                if self.yawn_frame_counter >= self.YAWN_CONSEC_FRAMES:
                    is_yawning = True
                    self.yawn_flag = True
                    self.total_yawns += 1
                    cv2.putText(frame, "YAWN DETECTED!", (10, 160),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            else:
                self.yawn_frame_counter = 0
                self.yawn_flag = False
            
            # Display MAR value near the face
            cv2.putText(frame, f"MAR: {mar_value:.2f}", (x, y + h + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        
        return frame, is_yawning, mar_value
    
    
    def detect_drowsiness_vit(self, frame):
        """
        Detect drowsiness using Vision Transformer via a closely-cropped face region.
        
        Args:
            frame: Input video frame
            
        Returns:
            drowsy_prob: Probability of drowsiness
            class_label: The raw class index
        """
        # Find the face first to avoid feeding the background into the model
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return 0.0, 2  # No face detected => Return 'open' (alert) as default to avoid false alarms
            
        # Get the largest face
        x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
        
        # Add a 20% margin just like the training dataset images
        margin_x = int(w * 0.2)
        margin_y = int(h * 0.2)
        
        y1 = max(0, y - margin_y)
        y2 = min(frame.shape[0], y + h + margin_y)
        x1 = max(0, x - margin_x)
        x2 = min(frame.shape[1], x + w + margin_x)
        
        # Crop the face from the frame
        face_img = frame[y1:y2, x1:x2]
        
        # Fallback if crop mathematically fails
        if face_img.size == 0:
            face_img = frame

        # Preprocess frame for ViT
        face_img = cv2.resize(face_img, (224, 224))
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_img = face_img.astype(np.float32) / 255.0
        
        # Normalize using ImageNet stats
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        face_img = (face_img - mean) / std
        
        # Convert to tensor and add batch dimension
        face_tensor = torch.from_numpy(face_img).permute(2, 0, 1).unsqueeze(0).float()
        face_tensor = face_tensor.to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.vit_model(face_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            
            # Dataset Classes (Alphabetical): 
            # 0: Active Subjects  (Alert)
            # 1: Fatigue Subjects (Drowsy)
            # 2: closed           (Drowsy)
            # 3: no_yawn          (Alert)
            # 4: open             (Alert)
            # 5: yawn             (Drowsy)
            
            prob_fatigue = probabilities[0][1].item()
            prob_closed = probabilities[0][2].item()
            prob_yawn = probabilities[0][5].item()
            
            # Total probability of being drowsy
            drowsy_prob = prob_fatigue + prob_closed + prob_yawn
            class_label = torch.argmax(probabilities, dim=1).item()
        
        return drowsy_prob, class_label
    
    
    def print_final_statistics(self):
        """Print comprehensive final statistics"""
        print("\n" + "="*60)
        print("FINAL DETECTION STATISTICS")
        print("="*60)
        
        if self.total_frames > 0:
            drowsy_percentage = (self.drowsy_frames / self.total_frames) * 100
            alert_percentage = (self.alert_frames / self.total_frames) * 100
            
            print(f"\nTotal Frames Processed: {self.total_frames}")
            print(f"Alert Frames: {self.alert_frames} ({alert_percentage:.2f}%)")
            print(f"Drowsy Frames: {self.drowsy_frames} ({drowsy_percentage:.2f}%)")
            
            if self.ear_history:
                print(f"\nEye Aspect Ratio Statistics:")
                print(f"  Average EAR: {np.mean(self.ear_history):.3f}")
                print(f"  Min EAR: {np.min(self.ear_history):.3f}")
                print(f"  Max EAR: {np.max(self.ear_history):.3f}")
                print(f"  Std Dev: {np.std(self.ear_history):.3f}")
            
            if self.mar_history:
                print(f"\nMouth Aspect Ratio Statistics:")
                print(f"  Average MAR: {np.mean(self.mar_history):.3f}")
                print(f"  Min MAR: {np.min(self.mar_history):.3f}")
                print(f"  Max MAR: {np.max(self.mar_history):.3f}")
                print(f"  Std Dev: {np.std(self.mar_history):.3f}")
                print(f"  Total Yawns: {self.total_yawns}")
            
            if self.vit_confidence_history:
                print(f"\nViT Confidence Statistics:")
                print(f"  Average Confidence: {np.mean(self.vit_confidence_history):.3f}")
                print(f"  Min Confidence: {np.min(self.vit_confidence_history):.3f}")
                print(f"  Max Confidence: {np.max(self.vit_confidence_history):.3f}")
            
            # Detection performance
            if drowsy_percentage > 50:
                print(f"\n⚠️  WARNING: High drowsiness detected ({drowsy_percentage:.1f}%)")
                print("   Consider taking a break!")
            elif drowsy_percentage > 20:
                print(f"\n⚠️  CAUTION: Moderate drowsiness detected ({drowsy_percentage:.1f}%)")
                print("   Stay alert!")
            else:
                print(f"\n✓  GOOD: Low drowsiness level ({drowsy_percentage:.1f}%)")
                print("   Keep up the good driving!")
        else:
            print("\nNo frames processed.")
        
        print("\n" + "="*60 + "\n")
    
    
    def draw_statistics_overlay(self, frame, ear_value, vit_prob, is_drowsy, fps,
                                    mar_value=0.0, is_yawning=False):
        """
        Draw statistics overlay on the frame
        
        Args:
            frame: Input frame
            ear_value: Current EAR value
            vit_prob: ViT drowsiness probability
            is_drowsy: Current drowsiness state
            fps: Current FPS
        """
        # Create semi-transparent overlay
        overlay = frame.copy()
        height, width = frame.shape[:2]
        
        # Stats panel dimensions (taller to fit yawn stats)
        panel_width = 300
        panel_height = 340
        panel_x = width - panel_width - 10
        panel_y = 10
        
        # Draw panel background
        cv2.rectangle(overlay, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height),
                     (0, 0, 0), -1)
        
        # Blend overlay
        alpha = 0.6
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        # Calculate statistics
        if self.total_frames > 0:
            drowsy_percentage = (self.drowsy_frames / self.total_frames) * 100
            alert_percentage = 100 - drowsy_percentage
            detection_rate = (self.alert_frames / self.total_frames) * 100
        else:
            drowsy_percentage = 0
            alert_percentage = 100
            detection_rate = 0
        
        # Store history for accuracy calculation
        if len(self.ear_history) > 100:
            self.ear_history.pop(0)
        if len(self.mar_history) > 100:
            self.mar_history.pop(0)
        if len(self.vit_confidence_history) > 100:
            self.vit_confidence_history.pop(0)
        
        self.ear_history.append(ear_value)
        self.mar_history.append(mar_value)
        self.vit_confidence_history.append(vit_prob)
        
        # Calculate average confidence
        avg_ear = np.mean(self.ear_history) if self.ear_history else 0
        avg_vit_conf = np.mean(self.vit_confidence_history) if self.vit_confidence_history else 0
        
        # Draw statistics text
        y_offset = panel_y + 25
        line_height = 30
        
        # Title
        cv2.putText(frame, "STATISTICS", (panel_x + 10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += line_height
        
        # FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (panel_x + 10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        y_offset += line_height
        
        # Current state with color
        state_text = "DROWSY" if is_drowsy else "ALERT"
        state_color = (0, 0, 255) if is_drowsy else (0, 255, 0)
        cv2.putText(frame, f"State: {state_text}", (panel_x + 10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, state_color, 2)
        y_offset += line_height
        
        # EAR value with bar
        cv2.putText(frame, f"EAR: {ear_value:.3f}", (panel_x + 10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        # EAR bar (normalized to 0-1)
        bar_length = int(min(ear_value / 0.4, 1.0) * 150)
        bar_color = (0, 255, 0) if ear_value > self.EAR_THRESHOLD else (0, 0, 255)
        cv2.rectangle(frame, (panel_x + 120, y_offset - 10), 
                     (panel_x + 120 + bar_length, y_offset), bar_color, -1)
        y_offset += line_height
        
        # ViT confidence with bar
        cv2.putText(frame, f"ViT Conf: {vit_prob:.3f}", (panel_x + 10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        # ViT confidence bar
        bar_length = int(vit_prob * 150)
        bar_color = (0, 0, 255) if vit_prob > 0.7 else (0, 255, 0)
        cv2.rectangle(frame, (panel_x + 120, y_offset - 10), 
                     (panel_x + 120 + bar_length, y_offset), bar_color, -1)
        y_offset += line_height
        
        # Detection accuracy
        cv2.putText(frame, f"Alert: {alert_percentage:.1f}%", (panel_x + 10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        y_offset += line_height
        
        cv2.putText(frame, f"Drowsy: {drowsy_percentage:.1f}%", (panel_x + 10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        y_offset += line_height
        
        # --- Yawn section ---
        y_offset += 5  # small gap
        cv2.line(frame, (panel_x + 10, y_offset - 5), (panel_x + panel_width - 10, y_offset - 5),
                 (100, 100, 100), 1)  # separator
        y_offset += line_height // 2
        
        # MAR value with bar
        cv2.putText(frame, f"MAR: {mar_value:.3f}", (panel_x + 10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        bar_length = int(min(mar_value / 1.0, 1.0) * 150)
        bar_color = (0, 165, 255) if mar_value > self.MAR_THRESHOLD else (0, 255, 0)
        cv2.rectangle(frame, (panel_x + 120, y_offset - 10),
                     (panel_x + 120 + bar_length, y_offset), bar_color, -1)
        y_offset += line_height
        
        # Yawn state
        yawn_text = "YAWNING" if is_yawning else "No Yawn"
        yawn_color = (0, 165, 255) if is_yawning else (0, 255, 0)
        cv2.putText(frame, f"Yawn: {yawn_text}", (panel_x + 10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, yawn_color, 1)
        y_offset += line_height
        
        # Total yawns count
        cv2.putText(frame, f"Total Yawns: {self.total_yawns}", (panel_x + 10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
        y_offset += line_height
        
        # Total frames
        cv2.putText(frame, f"Frames: {self.total_frames}", (panel_x + 10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    
    def run(self, video_source=0, use_vit=True, show_stats=True):
        """
        Run the drowsiness detection system
        
        Args:
            video_source: Camera index or video file path
            use_vit: Whether to use Vision Transformer in addition to EAR
            show_stats: Whether to show statistics overlay
        """
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            print("Error: Could not open video source")
            return
        
        print("Starting drowsiness detection...")
        print("Press 'q' to quit")
        print("Press 's' to toggle statistics overlay")
        
        fps_start_time = time.time()
        fps_counter = 0
        fps = 0
        current_ear = 0.0
        current_mar = 0.0
        current_vit_prob = 0.0
        is_yawning = False
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            self.total_frames += 1
            
            # EAR-based detection
            frame, ear_drowsy = self.detect_drowsiness_ear(frame)
            
            # Get current EAR value from the last detected face
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) > 0 and self.predictor is not None:
                x, y, w, h = faces[0]
                rect = dlib.rectangle(int(x), int(y), int(x+w), int(y+h))
                shape = self.predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)
                
                left_eye = shape[self.LEFT_EYE_START:self.LEFT_EYE_END]
                right_eye = shape[self.RIGHT_EYE_START:self.RIGHT_EYE_END]
                
                left_ear = eye_aspect_ratio(left_eye)
                right_ear = eye_aspect_ratio(right_eye)
                current_ear = (left_ear + right_ear) / 2.0
            
            # Yawn detection
            frame, is_yawning, current_mar = self.detect_yawn(frame)
            if is_yawning:
                self.yawn_frames += 1
            
            # ViT-based detection
            if use_vit:
                current_vit_prob, vit_class = self.detect_drowsiness_vit(frame)
                # Drowsy only if the model predicts it AND is highly confident (>70%)
                vit_drowsy = (vit_class in [0, 3]) and (current_vit_prob > 0.70)
            else:
                vit_drowsy = False
                current_vit_prob = 0.0
            
            # Combined decision (any method detects drowsiness / fatigue)
            is_drowsy = ear_drowsy or vit_drowsy or is_yawning
            
            if is_drowsy:
                self.drowsy_frames += 1
                self.drowsy_flag = True
                # Large warning text
                cv2.putText(frame, "*** DROWSINESS ALERT! ***", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                if is_yawning:
                    cv2.putText(frame, "*** YAWN DETECTED - STAY ALERT! ***", (10, 110),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 3)
                else:
                    cv2.putText(frame, "*** WAKE UP! ***", (10, 110),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
                
                # Play audio alarm (non-blocking)
                if time.time() - self.last_alarm_time > 1.0:
                    self.last_alarm_time = time.time()
                    try:
                        import winsound
                        import threading
                        # Beep at 2500Hz for 500ms
                        threading.Thread(target=lambda: winsound.Beep(2500, 500), daemon=True).start()
                    except ImportError:
                        pass
                        
            else:
                self.alert_frames += 1
            
            # Calculate and display FPS
            fps_counter += 1
            if (time.time() - fps_start_time) > 1:
                fps = fps_counter / (time.time() - fps_start_time)
                fps_counter = 0
                fps_start_time = time.time()
            
            # Draw statistics overlay
            if show_stats:
                frame = self.draw_statistics_overlay(frame, current_ear, current_vit_prob,
                                                     is_drowsy, fps, current_mar, is_yawning)
            else:
                # Just show FPS if stats are hidden
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display frame
            cv2.imshow('Driver Drowsiness Detection', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                show_stats = not show_stats
                print(f"Statistics overlay: {'ON' if show_stats else 'OFF'}")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Print final statistics
        self.print_final_statistics()


if __name__ == "__main__":
    # Initialize detector
    detector = DrowsinessDetector(use_cuda=True)
    
    # Run detection on webcam (0) or video file path
    detector.run(video_source=0, use_vit=True)
