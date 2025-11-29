"""
Deep Learning Model Architecture for Facial Emotion Recognition
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
)
import config
import logging

logger = logging.getLogger(__name__)


class EmotionRecognitionModel:
    """
    Convolutional Neural Network for Emotion Recognition
    """
    
    def __init__(self):
        """Initialize the model"""
        self.model = None
        self.history = None
    
    def build_model(self):
        """
        Build CNN architecture
        
        Returns:
            Compiled Keras model
        """
        logger.info("Building model architecture...")
        
        model = Sequential(name="EmotionRecognitionCNN")
        
        # Input shape
        input_shape = (config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS)
        
        # First Convolutional Block
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', 
                        input_shape=input_shape, padding='same', name='conv1'))
        model.add(BatchNormalization(name='bn1'))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', 
                        padding='same', name='conv2'))
        model.add(BatchNormalization(name='bn2'))
        model.add(MaxPooling2D(pool_size=config.POOL_SIZE, name='pool1'))
        model.add(Dropout(config.DROPOUT_CONV, name='dropout1'))
        
        # Second Convolutional Block
        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', 
                        padding='same', name='conv3'))
        model.add(BatchNormalization(name='bn3'))
        model.add(MaxPooling2D(pool_size=config.POOL_SIZE, name='pool2'))
        model.add(Dropout(config.DROPOUT_CONV, name='dropout2'))
        
        # Third Convolutional Block
        model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', 
                        padding='same', name='conv4'))
        model.add(BatchNormalization(name='bn4'))
        model.add(MaxPooling2D(pool_size=config.POOL_SIZE, name='pool3'))
        model.add(Dropout(config.DROPOUT_CONV, name='dropout3'))
        
        # Flatten and Dense Layers
        model.add(Flatten(name='flatten'))
        model.add(Dense(config.DENSE_UNITS, activation='relu', name='fc1'))
        model.add(BatchNormalization(name='bn5'))
        model.add(Dropout(config.DROPOUT_DENSE, name='dropout4'))
        
        model.add(Dense(256, activation='relu', name='fc2'))
        model.add(BatchNormalization(name='bn6'))
        model.add(Dropout(config.DROPOUT_DENSE, name='dropout5'))
        
        # Output Layer
        model.add(Dense(config.NUM_CLASSES, activation='softmax', name='output'))
        
        self.model = model
        logger.info("Model architecture built successfully")
        
        return model
    
    def compile_model(self, learning_rate=config.LEARNING_RATE):
        """
        Compile the model
        
        Args:
            learning_rate: Learning rate for optimizer
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        optimizer = Adam(learning_rate=learning_rate)
        
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info(f"Model compiled with Adam optimizer (lr={learning_rate})")
    
    def get_callbacks(self, model_path):
        """
        Get training callbacks
        
        Args:
            model_path: Path to save the best model
        
        Returns:
            List of callbacks
        """
        callbacks = [
            # Save best model
            ModelCheckpoint(
                filepath=model_path,
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            
            # Early stopping
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Reduce learning rate on plateau
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            
            # TensorBoard logging
            TensorBoard(
                log_dir=str(config.LOGS_DIR / 'tensorboard'),
                histogram_freq=1
            )
        ]
        
        logger.info("Callbacks configured")
        return callbacks
    
    def summary(self):
        """Print model summary"""
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        return self.model.summary()
    
    def get_model(self):
        """Get the Keras model"""
        return self.model
