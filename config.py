"""
Configuration file for Facial Emotion Recognition System
Contains all hyperparameters and settings
"""

import os
from pathlib import Path

# Project Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Ensure directories exist
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Model Configuration
IMG_HEIGHT = 48
IMG_WIDTH = 48
IMG_CHANNELS = 1  # Grayscale
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001

# Data Augmentation Parameters
ROTATION_RANGE = 30
SHEAR_RANGE = 0.3
ZOOM_RANGE = 0.3
HORIZONTAL_FLIP = True

# Model Architecture
CONV_LAYERS = [
    {'filters': 32, 'kernel_size': (3, 3), 'activation': 'relu'},
    {'filters': 64, 'kernel_size': (3, 3), 'activation': 'relu'},
    {'filters': 128, 'kernel_size': (3, 3), 'activation': 'relu'},
    {'filters': 256, 'kernel_size': (3, 3), 'activation': 'relu'}
]

POOL_SIZE = (2, 2)
DROPOUT_CONV = 0.1
DROPOUT_DENSE = 0.2
DENSE_UNITS = 512

# Emotion Labels
EMOTION_LABELS = {
    0: 'Angry',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Neutral',
    5: 'Sad',
    6: 'Surprise'
}

NUM_CLASSES = len(EMOTION_LABELS)

# Real-time Detection Configuration
CASCADE_PATH = str(BASE_DIR / "haarcascade_frontalface_default.xml")
CAMERA_INDEX = 0
SCALE_FACTOR = 1.3
MIN_NEIGHBORS = 5
MIN_FACE_SIZE = (30, 30)

# Display Configuration
WINDOW_NAME = "Facial Emotion Recognition"
FONT = 1  # cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.9
FONT_THICKNESS = 2
BOX_COLOR = (0, 255, 0)  # Green
TEXT_COLOR = (255, 255, 255)  # White
BOX_THICKNESS = 2

# Performance Optimization
SUPPRESS_TF_WARNINGS = True
GPU_MEMORY_GROWTH = True
