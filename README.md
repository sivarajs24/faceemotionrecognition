# Facial Emotion Recognition System

A professional deep learning system for real-time facial emotion recognition using Convolutional Neural Networks (CNN).

## Features

- **7 Emotion Classes**: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise
- **Real-time Detection**: Live emotion detection via webcam
- **Advanced CNN Architecture**: Deep learning model with BatchNormalization and Dropout
- **Data Augmentation**: Improved model generalization
- **Professional Logging**: Comprehensive logging system
- **Training Visualization**: Automatic plotting of training metrics
- **Model Callbacks**: Early stopping, learning rate reduction, model checkpointing
- **Configurable**: Centralized configuration management

## Project Structure

```
facialemotionalproject/
├── config.py              # Configuration and hyperparameters
├── model.py               # CNN model architecture
├── utils.py               # Utility functions
├── train.py               # Training script
├── detect.py              # Real-time detection script
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── data/
│   ├── train/            # Training data by emotion class
│   └── test/             # Test data by emotion class
├── models/               # Saved models
├── logs/                 # Training logs and screenshots
└── haarcascade_frontalface_default.xml  # Face detector

```

## Installation

1. **Clone or download the repository**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare your dataset**:
   - Place training images in `data/train/<emotion>/`
   - Place test images in `data/test/<emotion>/`
   - Emotion folders: angry, disgust, fear, happy, neutral, sad, surprise

## Usage

### Training the Model

```bash
python train.py
```

Features:
- Automatic data loading and augmentation
- Progress tracking with logging
- Model checkpointing (saves best model)
- Early stopping to prevent overfitting
- Learning rate reduction on plateau
- Training history visualization
- Comprehensive metrics logging

### Real-time Detection

```bash
python detect.py
```

Controls:
- Press **'q'** to quit
- Press **'s'** to save screenshot
- Face detection and emotion prediction in real-time
- Confidence scores displayed

## Configuration

Edit `config.py` to customize:

- **Model Parameters**: Learning rate, batch size, epochs
- **Architecture**: Layer sizes, dropout rates
- **Data Augmentation**: Rotation, zoom, shear ranges
- **Display Settings**: Colors, fonts, window size
- **Paths**: Data directories, model save locations

## Model Architecture

- **Input**: 48x48 grayscale images
- **Convolutional Blocks**: 4 blocks with BatchNormalization
  - 32, 64, 128, 256 filters
  - MaxPooling and Dropout layers
- **Dense Layers**: 2 fully connected layers
  - 512 and 256 units
  - BatchNormalization and Dropout
- **Output**: 7 classes with softmax activation

## Training Features

- **Data Augmentation**: Random rotation, shear, zoom, and flip
- **Callbacks**:
  - ModelCheckpoint: Save best model based on validation accuracy
  - EarlyStopping: Stop training if validation loss doesn't improve
  - ReduceLROnPlateau: Reduce learning rate when stuck
  - TensorBoard: Visualize training progress
- **Metrics Tracking**: Accuracy and loss for train/validation
- **Automatic Plotting**: Save training history graphs

## Results

After training, you'll find:
- **Best Model**: Saved in `models/` directory
- **Training Logs**: Detailed logs in `logs/` directory
- **Performance Plots**: Training/validation curves
- **Model Summary**: Architecture details

## Requirements

- Python 3.7+
- TensorFlow 2.x
- OpenCV
- NumPy
- Matplotlib
- Webcam (for real-time detection)

## Tips for Better Results

1. **Balanced Dataset**: Ensure similar number of images per emotion class
2. **Image Quality**: Use clear, well-lit face images
3. **Data Augmentation**: Already configured for better generalization
4. **Training Time**: More epochs may improve accuracy (adjust in config.py)
5. **GPU**: Training on GPU significantly faster than CPU

## Troubleshooting

**Webcam not detected**:
- Check camera permissions
- Adjust `CAMERA_INDEX` in config.py

**Model not loading**:
- Ensure model file exists in models/ directory
- Check file path in detect.py

**Low accuracy**:
- Increase training epochs
- Add more training data
- Adjust learning rate in config.py

## License

This project is for educational and research purposes.

## Author

Facial Emotion Recognition System - Professional Implementation

## Acknowledgments

- TensorFlow/Keras for deep learning framework
- OpenCV for computer vision capabilities
- Haar Cascade for face detection
