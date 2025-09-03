"""
Command-line argument utilities for the TibetanOCR project.
"""

import argparse
from .config import (
    DEFAULT_MODEL_PATH, DEFAULT_IMAGE_SIZE, DEFAULT_CONFIDENCE,
    DEFAULT_OUTPUT_DIR, DEFAULT_DATASET_DIR, DEFAULT_OCR_LANG,
    DEFAULT_TRAIN_SAMPLES, DEFAULT_VAL_SAMPLES, DEFAULT_AUGMENTATION,
    DEFAULT_FONT_PATH, DEFAULT_SBB_OUTPUT, DEFAULT_OCR_OUTPUT
)


def add_model_arguments(parser):
    """
    Add model-related arguments to an ArgumentParser.
    
    Args:
        parser: ArgumentParser instance
        
    Returns:
        ArgumentParser: Updated parser
    """
    group = parser.add_argument_group('Model Options')
    group.add_argument('--model', type=str, default=DEFAULT_MODEL_PATH,
                      help='Path to the model (e.g., yolov8n.pt, best.pt)')
    group.add_argument('--imgsz', type=int, default=DEFAULT_IMAGE_SIZE,
                      help='Image size for inference/training')
    group.add_argument('--conf', type=float, default=DEFAULT_CONFIDENCE,
                      help='Confidence threshold for detections')
    group.add_argument('--device', type=str, default='',
                      help='Device for inference/training (e.g., cpu, 0, 0,1,2,3)')
    return parser


def add_output_arguments(parser):
    """
    Add output-related arguments to an ArgumentParser.
    
    Args:
        parser: ArgumentParser instance
        
    Returns:
        ArgumentParser: Updated parser
    """
    group = parser.add_argument_group('Output Options')
    group.add_argument('--output', type=str, default=DEFAULT_OUTPUT_DIR,
                      help='Directory for output')
    group.add_argument('--name', type=str, default='exp',
                      help='Experiment name')
    group.add_argument('--save', action='store_true', default=True,
                      help='Save results')
    group.add_argument('--save-txt', action='store_true',
                      help='Save results as .txt files')
    group.add_argument('--save-conf', action='store_true',
                      help='Save confidence values in .txt files')
    return parser


def add_dataset_generation_arguments(parser):
    """
    Add dataset generation arguments to an ArgumentParser.
    
    Args:
        parser: ArgumentParser instance
        
    Returns:
        ArgumentParser: Updated parser
    """
    group = parser.add_argument_group('Dataset Generation Options')
    group.add_argument('--background_train', type=str, default='./data/background_images_train/',
                      help='Folder with background images for training')
    group.add_argument('--background_val', type=str, default='./data/background_images_val/',
                      help='Folder with background images for validation')
    group.add_argument('--dataset_name', type=str, default=DEFAULT_DATASET_DIR,
                      help='Folder for the generated YOLO dataset')
    group.add_argument('--corpora_folder', type=str, default='./data/corpora/Tibetan Number Words/',
                      help='Folder with Tibetan corpora')
    group.add_argument('--train_samples', type=int, default=DEFAULT_TRAIN_SAMPLES,
                      help='Number of training samples to generate')
    group.add_argument('--val_samples', type=int, default=DEFAULT_VAL_SAMPLES,
                      help='Number of validation samples to generate')
    group.add_argument('--no_cols', type=int, default=1,
                      help='Number of text columns to generate [1-5]')
    group.add_argument('--font_path', type=str, default=DEFAULT_FONT_PATH,
                      help='Path to a Tibetan font file')
    group.add_argument('--single_label', action='store_true',
                      help='Use a single label "tibetan" for all files')
    group.add_argument('--debug', action='store_true',
                      help='Enable debug mode for verbose output')
    group.add_argument('--image_size', type=int, default=DEFAULT_IMAGE_SIZE,
                      help='Size of generated images in pixels')
    group.add_argument('--augmentation', choices=['rotate', 'noise'], default=DEFAULT_AUGMENTATION,
                      help='Type of augmentation to apply')
    return parser


def add_training_arguments(parser):
    """
    Add training-related arguments to an ArgumentParser.
    
    Args:
        parser: ArgumentParser instance
        
    Returns:
        ArgumentParser: Updated parser
    """
    group = parser.add_argument_group('Training Options')
    group.add_argument('--dataset', type=str, default=DEFAULT_DATASET_DIR,
                      help='Name of the dataset folder')
    group.add_argument('--epochs', type=int, default=100,
                      help='Number of training epochs')
    group.add_argument('--batch', type=int, default=16,
                      help='Batch size for training')
    group.add_argument('--workers', type=int, default=8,
                      help='Number of workers for data loading')
    group.add_argument('--patience', type=int, default=50,
                      help='EarlyStopping patience in epochs')
    group.add_argument('--export', action='store_true',
                      help='Export the model after training as TorchScript')
    return parser


def add_wandb_arguments(parser):
    """
    Add Weights & Biases related arguments to an ArgumentParser.
    
    Args:
        parser: ArgumentParser instance
        
    Returns:
        ArgumentParser: Updated parser
    """
    group = parser.add_argument_group('Weights & Biases Options')
    group.add_argument('--wandb', action='store_true',
                      help='Enable Weights & Biases logging')
    group.add_argument('--wandb-project', type=str, default='TibetanOCR',
                      help='Weights & Biases project name')
    group.add_argument('--wandb-entity', type=str, default=None,
                      help='Weights & Biases entity (team or username)')
    group.add_argument('--wandb-tags', type=str, default=None,
                      help='Comma-separated tags for the experiment (e.g., "yolov8,tibetan")')
    group.add_argument('--wandb-name', type=str, default=None,
                      help='Name of the experiment in wandb (default: same as --name)')
    return parser


def add_sbb_arguments(parser):
    """
    Add Staatsbibliothek zu Berlin related arguments to an ArgumentParser.
    
    Args:
        parser: ArgumentParser instance
        
    Returns:
        ArgumentParser: Updated parser
    """
    group = parser.add_argument_group('SBB Options')
    group.add_argument('--ppn', type=str,
                      help='PPN (Pica Production Number) of the document in the Staatsbibliothek zu Berlin')
    group.add_argument('--download', action='store_true',
                      help='Download images instead of processing them directly')
    group.add_argument('--no-ssl-verify', action='store_true',
                      help='Disable SSL certificate verification (not recommended for production environments)')
    group.add_argument('--max-images', type=int, default=0,
                      help='Maximum number of images for inference (0 = all)')
    group.add_argument('--output_sbb_images', type=str, default=DEFAULT_SBB_OUTPUT,
                      help='Directory for saving downloaded images')
    return parser


def add_ocr_arguments(parser):
    """
    Add OCR-related arguments to an ArgumentParser.
    
    Args:
        parser: ArgumentParser instance
        
    Returns:
        ArgumentParser: Updated parser
    """
    group = parser.add_argument_group('OCR Options')
    group.add_argument('--lang', type=str, default=DEFAULT_OCR_LANG,
                      help='Language for Tesseract OCR (e.g., eng, deu, eng+deu, bod for Tibetan)')
    group.add_argument('--tesseract-config', type=str, default='',
                      help='Additional Tesseract configuration')
    group.add_argument('--save-crops', action='store_true',
                      help='Save cropped text blocks as images')
    group.add_argument('--ocr-output', type=str, default=DEFAULT_OCR_OUTPUT,
                      help='Directory for saving OCR results')
    return parser


def add_source_argument(parser):
    """
    Add source argument to an ArgumentParser.
    
    Args:
        parser: ArgumentParser instance
        
    Returns:
        ArgumentParser: Updated parser
    """
    parser.add_argument('--source', type=str,
                      help='Path to image or directory for inference')
    return parser


def create_generate_dataset_parser():
    """Create an ArgumentParser for dataset generation."""
    parser = argparse.ArgumentParser(description="Generate YOLO dataset for Tibetan text detection")
    parser = add_dataset_generation_arguments(parser)
    return parser


def create_train_parser():
    """Create an ArgumentParser for model training."""
    parser = argparse.ArgumentParser(description="Train a YOLO model with Tibetan OCR data")
    parser = add_model_arguments(parser)
    parser = add_training_arguments(parser)
    parser = add_output_arguments(parser)
    parser = add_wandb_arguments(parser)
    return parser


def create_inference_parser():
    """Create an ArgumentParser for inference."""
    parser = argparse.ArgumentParser(description="Run inference with a trained YOLO model")
    parser = add_model_arguments(parser)
    parser = add_output_arguments(parser)
    parser = add_source_argument(parser)
    return parser


def create_sbb_inference_parser():
    """Create an ArgumentParser for SBB inference."""
    parser = argparse.ArgumentParser(description="Run inference on Staatsbibliothek zu Berlin data")
    parser = add_model_arguments(parser)
    parser = add_output_arguments(parser)
    parser = add_sbb_arguments(parser)
    return parser


def create_ocr_parser():
    """Create an ArgumentParser for OCR on detected text blocks."""
    parser = argparse.ArgumentParser(description="Apply OCR to detected text blocks")
    parser = add_model_arguments(parser)
    parser = add_source_argument(parser)
    parser = add_sbb_arguments(parser)
    parser = add_ocr_arguments(parser)
    return parser
