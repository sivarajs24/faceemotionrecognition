"""
Real-time Facial Emotion Detection using Webcam - Enhanced Version
"""

import sys
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import logging
from pathlib import Path
import time

import config
from utils import setup_logging, suppress_tf_warnings


class EmotionDetector:
    """Real-time emotion detection from webcam feed with enhanced UI"""
    
    def __init__(self, model_path):
        """
        Initialize the emotion detector
        
        Args:
            model_path: Path to trained model file
        """
        self.logger = logging.getLogger(__name__)
        
        # Load model
        self.logger.info(f"Loading model from {model_path}")
        try:
            self.model = load_model(model_path)
            self.logger.info("Model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
        
        # Load face detector
        if not Path(config.CASCADE_PATH).exists():
            self.logger.error(f"Cascade file not found: {config.CASCADE_PATH}")
            raise FileNotFoundError(f"Cascade file not found: {config.CASCADE_PATH}")
        
        self.face_cascade = cv2.CascadeClassifier(config.CASCADE_PATH)
        self.logger.info("Face cascade classifier loaded")
        
        # Initialize webcam
        self.video = cv2.VideoCapture(config.CAMERA_INDEX)
        if not self.video.isOpened():
            self.logger.error("Failed to open webcam")
            raise RuntimeError("Failed to open webcam")
        
        # Set camera properties for better performance
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.video.set(cv2.CAP_PROP_FPS, 30)
        
        self.logger.info("Webcam initialized")
        
        # Emotion colors (BGR format) - vibrant colors
        self.emotion_colors = {
            'Angry': (0, 0, 255),       # Red
            'Disgust': (0, 200, 0),     # Green
            'Fear': (128, 0, 128),      # Purple
            'Happy': (0, 255, 255),     # Yellow
            'Neutral': (200, 200, 200), # Gray
            'Sad': (255, 100, 0),       # Blue
            'Surprise': (0, 165, 255)   # Orange
        }
        
        # Smooth predictions buffer for stability
        self.prediction_buffer = []
        self.buffer_size = 5
    
    def preprocess_face(self, face_img):
        """
        Preprocess face image for model prediction
        
        Args:
            face_img: Grayscale face image
        
        Returns:
            Preprocessed image array
        """
        # Resize to model input size
        resized = cv2.resize(face_img, (config.IMG_WIDTH, config.IMG_HEIGHT))
        
        # Normalize pixel values
        normalized = resized / 255.0
        
        # Reshape for model input
        reshaped = np.reshape(normalized, (1, config.IMG_HEIGHT, config.IMG_WIDTH, 1))
        
        return reshaped
    
    def predict_emotion(self, face_img):
        """
        Predict emotion from face image
        
        Args:
            face_img: Grayscale face image
        
        Returns:
            Tuple of (emotion_label, confidence, all_predictions)
        """
        # Preprocess
        processed_img = self.preprocess_face(face_img)
        
        # Predict
        predictions = self.model.predict(processed_img, verbose=0)
        
        # Get emotion with highest probability
        emotion_idx = np.argmax(predictions[0])
        confidence = predictions[0][emotion_idx]
        emotion_label = config.EMOTION_LABELS[emotion_idx]
        
        return emotion_label, confidence, predictions[0]
    
    def smooth_predictions(self, predictions):
        """Smooth predictions using a buffer to reduce flickering"""
        self.prediction_buffer.append(predictions)
        if len(self.prediction_buffer) > self.buffer_size:
            self.prediction_buffer.pop(0)
        
        # Average predictions
        avg_predictions = np.mean(self.prediction_buffer, axis=0)
        return avg_predictions
    
    def draw_face_box(self, frame, x, y, w, h, emotion, confidence):
        """
        Draw bounding box and emotion label on frame
        
        Args:
            frame: Video frame
            x, y, w, h: Face bounding box coordinates
            emotion: Predicted emotion label
            confidence: Prediction confidence
        """
        # Get color for this emotion
        color = self.emotion_colors.get(emotion, (0, 255, 0))
        
        # Draw main rectangle
        thickness = 2
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, thickness)
        
        # Draw corners for modern look
        corner_length = 15
        corner_thick = 3
        # Top-left
        cv2.line(frame, (x, y), (x + corner_length, y), color, corner_thick)
        cv2.line(frame, (x, y), (x, y + corner_length), color, corner_thick)
        # Top-right
        cv2.line(frame, (x+w, y), (x+w - corner_length, y), color, corner_thick)
        cv2.line(frame, (x+w, y), (x+w, y + corner_length), color, corner_thick)
        # Bottom-left
        cv2.line(frame, (x, y+h), (x + corner_length, y+h), color, corner_thick)
        cv2.line(frame, (x, y+h), (x, y+h - corner_length), color, corner_thick)
        # Bottom-right
        cv2.line(frame, (x+w, y+h), (x+w - corner_length, y+h), color, corner_thick)
        cv2.line(frame, (x+w, y+h), (x+w, y+h - corner_length), color, corner_thick)
        
        # Prepare text
        text = f"{emotion}"
        confidence_text = f"{confidence*100:.1f}%"
        
        # Get text sizes
        font_scale = 0.8
        font_thickness = 2
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, font_scale, font_thickness)
        (conf_width, conf_height), _ = cv2.getTextSize(confidence_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
        # Draw background for text
        padding = 8
        overlay = frame.copy()
        cv2.rectangle(overlay, 
                     (x, y - text_height - conf_height - padding * 3),
                     (x + max(text_width, conf_width) + padding * 2, y),
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw emotion text
        cv2.putText(frame, text, (x + padding, y - conf_height - padding * 2),
                   cv2.FONT_HERSHEY_DUPLEX, font_scale, color, font_thickness)
        
        # Draw confidence
        cv2.putText(frame, confidence_text, (x + padding, y - padding),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def run(self):
        """Start real-time emotion detection"""
        self.logger.info("Starting real-time emotion detection")
        self.logger.info("Press 'q' to quit, 's' to save screenshot, 'r' to reset")
        
        frame_count = 0
        screenshot_count = 0
        fps_counter = []
        
        try:
            while True:
                start_time = time.time()
                
                # Read frame
                ret, frame = self.video.read()
                
                if not ret:
                    self.logger.error("Failed to read frame from webcam")
                    break
                
                frame_count += 1
                
                # Flip frame for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Apply slight blur for better face detection
                frame_blur = cv2.GaussianBlur(frame, (3, 3), 0)
                
                # Convert to grayscale
                gray = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2GRAY)
                
                # Apply histogram equalization
                gray = cv2.equalizeHist(gray)
                
                # Detect faces
                faces = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(50, 50),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                
                # Process each detected face
                for (x, y, w, h) in faces:
                    # Extract face region with padding
                    padding = int(0.1 * w)
                    x_pad = max(0, x - padding)
                    y_pad = max(0, y - padding)
                    w_pad = min(frame.shape[1] - x_pad, w + 2 * padding)
                    h_pad = min(frame.shape[0] - y_pad, h + 2 * padding)
                    
                    face_roi = gray[y_pad:y_pad+h_pad, x_pad:x_pad+w_pad]
                    
                    # Predict emotion
                    emotion, confidence, all_preds = self.predict_emotion(face_roi)
                    
                    # Smooth predictions
                    smoothed_preds = self.smooth_predictions(all_preds)
                    emotion_idx = np.argmax(smoothed_preds)
                    emotion = config.EMOTION_LABELS[emotion_idx]
                    confidence = smoothed_preds[emotion_idx]
                    
                    # Draw on frame
                    self.draw_face_box(frame, x, y, w, h, emotion, confidence)
                
                # Calculate FPS
                fps = 1.0 / (time.time() - start_time)
                fps_counter.append(fps)
                if len(fps_counter) > 30:
                    fps_counter.pop(0)
                avg_fps = np.mean(fps_counter)
                
                # Create HUD overlay
                overlay = frame.copy()
                hud_height = 50
                cv2.rectangle(overlay, (0, 0), (frame.shape[1], hud_height), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
                
                # Add info text
                info_text = f"Faces: {len(faces)}  |  FPS: {avg_fps:.1f}  |  Frame: {frame_count}"
                cv2.putText(frame, info_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Add controls hint
                controls = "Q: Quit | S: Screenshot | R: Reset"
                cv2.putText(frame, controls, (10, frame.shape[0] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                
                # Show frame
                cv2.imshow("Facial Emotion Recognition", frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    self.logger.info("Quit command received")
                    break
                elif key == ord('s'):
                    screenshot_path = config.LOGS_DIR / f"screenshot_{screenshot_count}.jpg"
                    cv2.imwrite(str(screenshot_path), frame)
                    screenshot_count += 1
                    self.logger.info(f"Screenshot saved: {screenshot_path}")
                elif key == ord('r'):
                    self.prediction_buffer = []
                    self.logger.info("Prediction buffer reset")
        
        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")
        
        except Exception as e:
            self.logger.error(f"Error during detection: {e}", exc_info=True)
        
        finally:
            # Cleanup
            self.logger.info("Cleaning up resources")
            self.video.release()
            cv2.destroyAllWindows()
            self.logger.info(f"Total frames processed: {frame_count}")


def main():
    """Main function"""
    # Suppress TensorFlow warnings
    if config.SUPPRESS_TF_WARNINGS:
        suppress_tf_warnings()
    
    # Setup logging
    logger = setup_logging(str(config.LOGS_DIR), "detection")
    
    # Find model file
    model_path = None
    
    # Check for specific model file
    if Path("model_file_30epochs.h5").exists():
        model_path = "model_file_30epochs.h5"
    elif (config.MODELS_DIR / "emotion_model_final.h5").exists():
        model_path = config.MODELS_DIR / "emotion_model_final.h5"
    else:
        # Find latest model in models directory
        model_files = list(config.MODELS_DIR.glob("*.h5"))
        if model_files:
            model_path = max(model_files, key=lambda p: p.stat().st_mtime)
    
    if model_path is None:
        logger.error("No trained model found. Please train the model first.")
        sys.exit(1)
    
    logger.info(f"Using model: {model_path}")
    
    # Start detection
    detector = EmotionDetector(str(model_path))
    detector.run()


if __name__ == "__main__":
    main()
