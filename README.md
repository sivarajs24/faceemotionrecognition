# 🎭 Facial Emotion Recognition System

A real-time facial emotion recognition system using deep learning (CNN) with a professional web interface. Detects and classifies 7 human emotions from live webcam feeds.

## ✨ Features

- **Real-time Emotion Detection**: Live webcam-based emotion classification
- **7 Emotion Classes**: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise
- **Web-Based Interface**: Professional UI with gradient design and live statistics
- **Multi-Face Detection**: Detect emotions on multiple faces simultaneously
- **Live Analytics Dashboard**: Real-time emotion distribution with percentages
- **Camera Controls**: Start/Stop/Screenshot buttons for user control
- **Advanced CNN Model**: 5.2M parameters with BatchNormalization for stability
- **Prediction Smoothing**: 5-frame buffer to reduce false positives
- **Preprocessing Pipeline**: Histogram equalization and Gaussian blur for robustness

## 🏗️ Project Structure

```
faceemotionrecognition/
├── app.py                      # Flask web application
├── config.py                   # Configuration & hyperparameters
├── model.py                    # CNN architecture
├── train.py                    # Training script
├── detect.py                   # Desktop detection (optional)
├── utils.py                    # Utility functions
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── .gitignore                 # Git ignore patterns
├── haarcascade_frontalface_default.xml  # Face detector
├── model_file_30epochs.h5     # Trained model
├── templates/
│   └── index.html             # Web interface
└── data/
    ├── train/                 # Training images
    │   ├── angry/
    │   ├── disgust/
    │   ├── fear/
    │   ├── happy/
    │   ├── neutral/
    │   ├── sad/
    │   └── surprise/
    └── test/                  # Test images
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Webcam (for real-time detection)
- 500MB disk space

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/sivarajs24/faceemotionrecognition.git
cd faceemotionrecognition
```

2. **Create virtual environment** (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Running the Application

#### Web-Based Emotion Recognition (Recommended)
```bash
python app.py
```
Then open your browser and go to: **http://localhost:5000**

**Features:**
- Click **Start** to begin emotion detection
- Watch live emotion labels and confidence scores
- Real-time statistics update every 2 seconds
- Click **Stop** to pause detection
- **Screenshot** button to capture current frame
- **Reset** button to clear statistics

#### Desktop Detection (Real-time)
```bash
python detect.py
```
Press `q` to quit the application.

#### Training the Model
```bash
python train.py
```
Customize training parameters in `config.py`:
- `EPOCHS`: Number of training epochs
- `BATCH_SIZE`: Batch size for training
- `LEARNING_RATE`: Learning rate for optimizer

## 📊 Model Architecture

**CNN with 5.2M Parameters:**
- **Block 1**: Conv2D(32) → Conv2D(64) → MaxPool → Dropout
- **Block 2**: Conv2D(128) → MaxPool → Dropout
- **Block 3**: Conv2D(256) → MaxPool → Dropout
- **Fully Connected**: Dense(512) → Dense(256) → Dense(7)
- **Normalization**: BatchNormalization after each Conv2D layer

**Training Details:**
- Dataset: 35,000+ facial images
- Image Size: 48×48 grayscale
- Optimizer: Adam with learning rate decay
- Loss Function: Categorical crossentropy
- Validation Split: 20%

## 🛠️ Technical Stack

| Component | Technology |
|-----------|-----------|
| **Backend** | Flask 3.1.2 |
| **Deep Learning** | TensorFlow 2.x, Keras |
| **Computer Vision** | OpenCV (cv2) |
| **Data Processing** | NumPy, Pillow |
| **Frontend** | HTML5, CSS3, JavaScript |
| **Visualization** | Matplotlib |
| **Version Control** | Git |

## 📈 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web interface |
| `/video_feed` | GET | MJPEG video stream |
| `/stats` | GET | Emotion statistics (JSON) |
| `/start_camera` | GET | Activate camera |
| `/stop_camera` | GET | Pause camera |
| `/reset_stats` | GET | Reset statistics |
| `/camera_status` | GET | Get camera status |

## 📸 Screenshots

The web interface features:
- Live video feed with emotion labels
- Real-time statistics panel
- Animated emotion distribution bars
- Professional gradient design
- Responsive layout for all devices

## 🧠 How It Works

1. **Face Detection**: Haar Cascade Classifier detects faces in video stream
2. **Preprocessing**: 
   - Grayscale conversion
   - Histogram equalization for better contrast
   - Gaussian blur for noise reduction
3. **Emotion Prediction**: CNN model predicts emotion probability
4. **Smoothing**: 5-frame buffer averages predictions
5. **Display**: Emotion label and confidence score shown on face
6. **Statistics**: Emotion counts and percentages tracked in real-time

## 🎯 Performance

- **Inference Speed**: ~20-30 FPS on CPU
- **Model Size**: 20MB (H5 format)
- **Face Detection Accuracy**: High with preprocessing
- **Emotion Classification**: Optimized for 7 classes

## 📚 Configuration

Edit `config.py` to customize:

```python
# Model
IMG_HEIGHT = 48
IMG_WIDTH = 48
EPOCHS = 30
BATCH_SIZE = 32

# Face Detection
SCALE_FACTOR = 1.1
MIN_NEIGHBORS = 5
MIN_FACE_SIZE = (50, 50)

# Emotion Classes
EMOTION_LABELS = {
    0: 'Angry', 1: 'Disgust', 2: 'Fear',
    3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'
}
```

## 🔧 Troubleshooting

**Camera not found:**
- Ensure webcam permissions are granted
- Check camera is not in use by another application

**Low FPS:**
- Reduce frame resolution in `config.py`
- Close unnecessary applications

**Inaccurate detections:**
- Ensure adequate lighting
- Move closer to camera
- Update model with more training data

## 📝 Dataset Structure

```
data/
├── train/
│   ├── angry/       (4,000+ images)
│   ├── disgust/     (4,000+ images)
│   ├── fear/        (4,000+ images)
│   ├── happy/       (5,000+ images)
│   ├── neutral/     (5,000+ images)
│   ├── sad/         (5,000+ images)
│   └── surprise/    (4,000+ images)
└── test/
    └── (Similar structure with validation images)
```

## 📄 Requirements

```
tensorflow>=2.13.0
keras>=2.13.0
opencv-python>=4.8.0
numpy>=1.23.0
flask>=3.1.2
pillow>=10.0.0
matplotlib>=3.8.0
```

## 🚀 Future Enhancements

- [ ] Multi-person emotion tracking with unique IDs
- [ ] Emotion timeline visualization
- [ ] Cloud deployment (AWS/Google Cloud)
- [ ] Mobile app integration
- [ ] Real-time audio emotion detection
- [ ] Emotion intensity measurement
- [ ] Cross-cultural emotion models
- [ ] API for third-party integration

## 📖 Learning Resources

- [TensorFlow Documentation](https://www.tensorflow.org/learn)
- [OpenCV Face Detection](https://docs.opencv.org/master/db/d28/tutorial_cascade_classifier.html)
- [Flask Web Development](https://flask.palletsprojects.com/)
- [CNN Architecture Guide](https://en.wikipedia.org/wiki/Convolutional_neural_network)

## 📝 License

This project is open source and available under the MIT License.

## 👨‍💻 Author

**Sivaraj S**
- GitHub: [@sivarajs24](https://github.com/sivarajs24)
- Email: harishsiva242005@gmail.com

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## ⭐ Support

If you find this project useful, please consider giving it a star! ⭐

---

**Last Updated**: December 2025
**Version**: 1.0.0
