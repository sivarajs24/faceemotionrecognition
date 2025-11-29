"""
Training script for Facial Emotion Recognition Model
"""

import os
import sys
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import config
from model import EmotionRecognitionModel
from utils import (
    setup_logging, configure_gpu, suppress_tf_warnings,
    plot_training_history, print_class_distribution, save_model_info
)


def create_data_generators():
    """
    Create training and validation data generators
    
    Returns:
        train_generator, validation_generator
    """
    logger.info("Setting up data generators...")
    
    # Training data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=config.ROTATION_RANGE,
        shear_range=config.SHEAR_RANGE,
        zoom_range=config.ZOOM_RANGE,
        horizontal_flip=config.HORIZONTAL_FLIP,
        fill_mode='nearest'
    )
    
    # Validation data (only rescaling)
    validation_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create generators
    train_generator = train_datagen.flow_from_directory(
        str(config.TRAIN_DIR),
        color_mode='grayscale',
        target_size=(config.IMG_HEIGHT, config.IMG_WIDTH),
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )
    
    validation_generator = validation_datagen.flow_from_directory(
        str(config.TEST_DIR),
        color_mode='grayscale',
        target_size=(config.IMG_HEIGHT, config.IMG_WIDTH),
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    logger.info(f"Found {train_generator.samples} training images")
    logger.info(f"Found {validation_generator.samples} validation images")
    logger.info(f"Class indices: {train_generator.class_indices}")
    
    return train_generator, validation_generator


def train_model():
    """Main training function"""
    
    # Setup
    logger.info("="*70)
    logger.info("FACIAL EMOTION RECOGNITION - TRAINING")
    logger.info("="*70)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Print dataset information
    print_class_distribution(str(config.TRAIN_DIR), str(config.TEST_DIR), logger)
    
    # Create data generators
    train_generator, validation_generator = create_data_generators()
    
    # Build and compile model
    emotion_model = EmotionRecognitionModel()
    emotion_model.build_model()
    emotion_model.compile_model()
    
    # Print model summary
    logger.info("\n" + "="*70)
    logger.info("MODEL ARCHITECTURE")
    logger.info("="*70)
    emotion_model.summary()
    
    # Save model architecture
    model_summary_path = config.MODELS_DIR / "model_architecture.txt"
    save_model_info(emotion_model.get_model(), str(model_summary_path))
    
    # Setup callbacks
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = config.MODELS_DIR / f"emotion_model_{timestamp}.h5"
    callbacks = emotion_model.get_callbacks(str(model_path))
    
    # Calculate steps
    steps_per_epoch = train_generator.samples // config.BATCH_SIZE
    validation_steps = validation_generator.samples // config.BATCH_SIZE
    
    logger.info("\n" + "="*70)
    logger.info("TRAINING CONFIGURATION")
    logger.info("="*70)
    logger.info(f"Epochs: {config.EPOCHS}")
    logger.info(f"Batch size: {config.BATCH_SIZE}")
    logger.info(f"Steps per epoch: {steps_per_epoch}")
    logger.info(f"Validation steps: {validation_steps}")
    logger.info(f"Learning rate: {config.LEARNING_RATE}")
    logger.info("="*70 + "\n")
    
    # Train model
    logger.info("Starting training...")
    
    try:
        history = emotion_model.get_model().fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=config.EPOCHS,
            validation_data=validation_generator,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save final model
        final_model_path = config.MODELS_DIR / "emotion_model_final.h5"
        emotion_model.get_model().save(str(final_model_path))
        logger.info(f"Final model saved to {final_model_path}")
        
        # Plot training history
        plot_path = config.LOGS_DIR / f"training_history_{timestamp}.png"
        plot_training_history(history, str(plot_path))
        
        # Print final metrics
        logger.info("\n" + "="*70)
        logger.info("TRAINING COMPLETED")
        logger.info("="*70)
        logger.info(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
        logger.info(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
        logger.info(f"Final training loss: {history.history['loss'][-1]:.4f}")
        logger.info(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")
        logger.info(f"Best model saved to: {model_path}")
        logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*70)
        
    except KeyboardInterrupt:
        logger.warning("\nTraining interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    # Suppress warnings
    if config.SUPPRESS_TF_WARNINGS:
        suppress_tf_warnings()
    
    # Configure GPU
    if config.GPU_MEMORY_GROWTH:
        configure_gpu()
    
    # Setup logging
    logger = setup_logging(str(config.LOGS_DIR), "training")
    
    # Start training
    train_model()
