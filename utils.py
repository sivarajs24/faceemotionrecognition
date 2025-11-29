"""
Utility functions for Facial Emotion Recognition System
"""

import os
import logging
from datetime import datetime
from pathlib import Path
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def setup_logging(log_dir="logs", name="emotion_recognition"):
    """
    Setup logging configuration
    
    Args:
        log_dir: Directory to save log files
        name: Name of the logger
    
    Returns:
        Logger instance
    """
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(name)


def configure_gpu():
    """Configure GPU settings for optimal performance"""
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logging.info(f"GPU(s) detected: {len(gpus)}")
        else:
            logging.info("No GPU detected. Using CPU.")
    except Exception as e:
        logging.warning(f"GPU configuration failed: {e}")


def suppress_tf_warnings():
    """Suppress TensorFlow warnings"""
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.get_logger().setLevel('ERROR')


def count_images(directory):
    """
    Count total images in directory and subdirectories
    
    Args:
        directory: Path to directory
    
    Returns:
        Total number of image files
    """
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    count = 0
    
    for root, dirs, files in os.walk(directory):
        count += sum(1 for f in files if Path(f).suffix.lower() in valid_extensions)
    
    return count


def plot_training_history(history, save_path=None):
    """
    Plot training and validation accuracy/loss
    
    Args:
        history: Keras History object
        save_path: Path to save the plot (optional)
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy plot
    axes[0].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Loss plot
    axes[1].plot(history.history['loss'], label='Training Loss')
    axes[1].plot(history.history['val_loss'], label='Validation Loss')
    axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Training history plot saved to {save_path}")
    
    plt.show()


def get_class_distribution(directory):
    """
    Get distribution of classes in dataset
    
    Args:
        directory: Path to dataset directory
    
    Returns:
        Dictionary with class names and counts
    """
    distribution = {}
    
    for class_dir in os.listdir(directory):
        class_path = os.path.join(directory, class_dir)
        if os.path.isdir(class_path):
            count = len([f for f in os.listdir(class_path) 
                        if os.path.isfile(os.path.join(class_path, f))])
            distribution[class_dir] = count
    
    return distribution


def print_class_distribution(train_dir, test_dir, logger=None):
    """
    Print dataset distribution information
    
    Args:
        train_dir: Training data directory
        test_dir: Test data directory
        logger: Logger instance (optional)
    """
    log_func = logger.info if logger else print
    
    train_dist = get_class_distribution(train_dir)
    test_dist = get_class_distribution(test_dir)
    
    log_func("\n" + "="*60)
    log_func("DATASET DISTRIBUTION")
    log_func("="*60)
    
    log_func(f"\n{'Class':<15} {'Train':<10} {'Test':<10} {'Total':<10}")
    log_func("-"*60)
    
    total_train = 0
    total_test = 0
    
    for emotion in sorted(train_dist.keys()):
        train_count = train_dist.get(emotion, 0)
        test_count = test_dist.get(emotion, 0)
        total_train += train_count
        total_test += test_count
        log_func(f"{emotion:<15} {train_count:<10} {test_count:<10} {train_count + test_count:<10}")
    
    log_func("-"*60)
    log_func(f"{'TOTAL':<15} {total_train:<10} {total_test:<10} {total_train + total_test:<10}")
    log_func("="*60 + "\n")


def save_model_info(model, filepath):
    """
    Save model architecture summary to file
    
    Args:
        model: Keras model
        filepath: Path to save the summary
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    
    logging.info(f"Model architecture saved to {filepath}")
