"""
Flask Web Application for Facial Emotion Recognition
"""

from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import logging
from pathlib import Path
import time
import base64
import json

import config
from utils import suppress_tf_warnings

# Suppress TensorFlow warnings
if config.SUPPRESS_TF_WARNINGS:
    suppress_tf_warnings()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'

# Global variables
camera = None
model = None
face_cascade = None
emotion_stats = {emotion: 0 for emotion in config.EMOTION_LABELS.values()}
total_detections = 0
prediction_buffer = []
BUFFER_SIZE = 5
camera_active = False

# Emotion colors (RGB for web)
EMOTION_COLORS = {
    'Angry': '#FF0000',
    'Disgust': '#00C800',
    'Fear': '#800080',
    'Happy': '#FFFF00',
    'Neutral': '#C8C8C8',
    'Sad': '#FF6400',
    'Surprise': '#FFA500'
}


def load_resources():
    """Load model and face cascade"""
    global model, face_cascade
    
    # Load model
    model_path = None
    if Path("model_file_30epochs.h5").exists():
        model_path = "model_file_30epochs.h5"
    elif (config.MODELS_DIR / "emotion_model_final.h5").exists():
        model_path = config.MODELS_DIR / "emotion_model_final.h5"
    else:
        model_files = list(config.MODELS_DIR.glob("*.h5"))
        if model_files:
            model_path = max(model_files, key=lambda p: p.stat().st_mtime)
    
    if model_path is None:
        raise FileNotFoundError("No trained model found")
    
    model = load_model(str(model_path))
    print(f"Model loaded: {model_path}")
    
    # Load face cascade
    if not Path(config.CASCADE_PATH).exists():
        raise FileNotFoundError(f"Cascade file not found: {config.CASCADE_PATH}")
    
    face_cascade = cv2.CascadeClassifier(config.CASCADE_PATH)
    print("Face cascade loaded")


def get_camera():
    """Get camera instance"""
    global camera
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return camera


def preprocess_face(face_img):
    """Preprocess face for prediction"""
    resized = cv2.resize(face_img, (config.IMG_WIDTH, config.IMG_HEIGHT))
    normalized = resized / 255.0
    reshaped = np.reshape(normalized, (1, config.IMG_HEIGHT, config.IMG_WIDTH, 1))
    return reshaped


def predict_emotion(face_img):
    """Predict emotion from face image"""
    processed_img = preprocess_face(face_img)
    predictions = model.predict(processed_img, verbose=0)
    emotion_idx = np.argmax(predictions[0])
    confidence = predictions[0][emotion_idx]
    emotion_label = config.EMOTION_LABELS[emotion_idx]
    return emotion_label, confidence, predictions[0]


def smooth_predictions(predictions):
    """Smooth predictions using buffer"""
    global prediction_buffer
    prediction_buffer.append(predictions)
    if len(prediction_buffer) > BUFFER_SIZE:
        prediction_buffer.pop(0)
    avg_predictions = np.mean(prediction_buffer, axis=0)
    return avg_predictions


def generate_frames():
    """Generate video frames with emotion detection"""
    global emotion_stats, total_detections, camera_active
    
    cam = get_camera()
    
    while True:
        if not camera_active:
            # Create a blank frame when camera is off
            blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank_frame, "Camera Stopped", (180, 240),
                       cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 255), 2)
            cv2.putText(blank_frame, "Click 'Start Camera' to begin", (140, 280),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 1)
            ret, buffer = cv2.imencode('.jpg', blank_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            continue
        success, frame = cam.read()
        if not success:
            break
        
        # Flip for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Prepare for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50)
        )
        
        # Process each face
        for (x, y, w, h) in faces:
            # Extract face with padding
            padding = int(0.1 * w)
            x_pad = max(0, x - padding)
            y_pad = max(0, y - padding)
            w_pad = min(frame.shape[1] - x_pad, w + 2 * padding)
            h_pad = min(frame.shape[0] - y_pad, h + 2 * padding)
            
            face_roi = gray[y_pad:y_pad+h_pad, x_pad:x_pad+w_pad]
            
            # Predict emotion
            emotion, confidence, all_preds = predict_emotion(face_roi)
            
            # Smooth predictions
            smoothed_preds = smooth_predictions(all_preds)
            emotion_idx = np.argmax(smoothed_preds)
            emotion = config.EMOTION_LABELS[emotion_idx]
            confidence = smoothed_preds[emotion_idx]
            
            # Update statistics
            emotion_stats[emotion] += 1
            total_detections += 1
            
            # Draw rectangle
            color = (0, 255, 0)
            if emotion == 'Angry':
                color = (0, 0, 255)
            elif emotion == 'Happy':
                color = (0, 255, 255)
            elif emotion == 'Sad':
                color = (255, 100, 0)
            elif emotion == 'Surprise':
                color = (0, 165, 255)
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Draw corners
            corner_length = 15
            cv2.line(frame, (x, y), (x + corner_length, y), color, 3)
            cv2.line(frame, (x, y), (x, y + corner_length), color, 3)
            cv2.line(frame, (x+w, y), (x+w - corner_length, y), color, 3)
            cv2.line(frame, (x+w, y), (x+w, y + corner_length), color, 3)
            cv2.line(frame, (x, y+h), (x + corner_length, y+h), color, 3)
            cv2.line(frame, (x, y+h), (x, y+h - corner_length), color, 3)
            cv2.line(frame, (x+w, y+h), (x+w - corner_length, y+h), color, 3)
            cv2.line(frame, (x+w, y+h), (x+w, y+h - corner_length), color, 3)
            
            # Add text
            text = f"{emotion}"
            conf_text = f"{confidence*100:.1f}%"
            
            cv2.putText(frame, text, (x, y - 30),
                       cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 2)
            cv2.putText(frame, conf_text, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stats')
def stats():
    """Get emotion statistics"""
    global emotion_stats, total_detections
    
    # Define colors for each emotion
    emotion_colors = {
        'Angry': '#e74c3c',
        'Disgust': '#8e44ad',
        'Fear': '#2c3e50',
        'Happy': '#f39c12',
        'Neutral': '#95a5a6',
        'Sad': '#3498db',
        'Surprise': '#e67e22'
    }
    
    stats_data = {
        'emotions': {},
        'total': total_detections
    }
    
    # Calculate percentages and prepare emotion data
    for emotion, count in emotion_stats.items():
        percentage = (count / total_detections * 100) if total_detections > 0 else 0
        stats_data['emotions'][emotion] = {
            'count': count,
            'percentage': round(percentage, 1),
            'color': emotion_colors.get(emotion, '#95a5a6')
        }
    
    return jsonify(stats_data)


@app.route('/reset_stats')
def reset_stats():
    """Reset statistics"""
    global emotion_stats, total_detections, prediction_buffer
    emotion_stats = {emotion: 0 for emotion in config.EMOTION_LABELS.values()}
    total_detections = 0
    prediction_buffer = []
    return jsonify({'status': 'success'})


@app.route('/start_camera')
def start_camera():
    """Start camera"""
    global camera_active
    camera_active = True
    return jsonify({'status': 'success', 'active': True})


@app.route('/stop_camera')
def stop_camera():
    """Stop camera"""
    global camera_active
    camera_active = False
    return jsonify({'status': 'success', 'active': False})


@app.route('/camera_status')
def camera_status():
    """Get camera status"""
    global camera_active
    return jsonify({'active': camera_active})


if __name__ == '__main__':
    print("Loading resources...")
    load_resources()
    print("Starting web server...")
    print("Open your browser and go to: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
