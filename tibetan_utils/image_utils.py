"""
Image processing utilities for the TibetanOCR project.
"""

import cv2
import numpy as np
from PIL import Image
import io
from typing import Tuple, List, Union, Dict, Any


def load_image(image_path: str) -> np.ndarray:
    """
    Load an image from a file path.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        np.ndarray: Image as a numpy array
    """
    return cv2.imread(image_path)


def pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
    """
    Convert a PIL Image to a cv2 image (numpy array).
    
    Args:
        pil_image: PIL Image
        
    Returns:
        np.ndarray: cv2 image
    """
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def cv2_to_pil(cv2_image: np.ndarray) -> Image.Image:
    """
    Convert a cv2 image (numpy array) to a PIL Image.
    
    Args:
        cv2_image: cv2 image
        
    Returns:
        Image.Image: PIL Image
    """
    return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))


def resize_image(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """
    Resize an image.
    
    Args:
        image: Image to resize
        size: Target size (width, height)
        
    Returns:
        np.ndarray: Resized image
    """
    return cv2.resize(image, size)


def extract_text_region(image: np.ndarray, box: List[float]) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Extract a text region from an image based on a bounding box.
    
    Args:
        image: Image as a numpy array
        box: Bounding box [x, y, w, h, conf, class] (normalized coordinates)
        
    Returns:
        Tuple[np.ndarray, Tuple[int, int, int, int]]: Extracted region and absolute coordinates (x_min, y_min, x_max, y_max)
    """
    # Extract coordinates
    x, y, w, h = box[:4]
    
    # Convert relative coordinates to absolute pixel coordinates
    height, width = image.shape[:2]
    x_min = int((x - w/2) * width)
    y_min = int((y - h/2) * height)
    x_max = int((x + w/2) * width)
    y_max = int((y + h/2) * height)
    
    # Ensure coordinates are within image bounds
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(width, x_max)
    y_max = min(height, y_max)
    
    # Extract the region
    region = image[y_min:y_max, x_min:x_max]
    
    return region, (x_min, y_min, x_max, y_max)


def draw_boxes(image: np.ndarray, boxes: List[List[float]], labels: List[str] = None, 
               colors: List[Tuple[int, int, int]] = None) -> np.ndarray:
    """
    Draw bounding boxes on an image.
    
    Args:
        image: Image to draw on
        boxes: List of bounding boxes [x, y, w, h, conf, class]
        labels: List of labels for each box
        colors: List of colors for each box
        
    Returns:
        np.ndarray: Image with boxes drawn
    """
    result = image.copy()
    
    if colors is None:
        colors = [(0, 255, 0)] * len(boxes)  # Default to green
    
    for i, box in enumerate(boxes):
        x, y, w, h = box[:4]
        
        # Convert relative coordinates to absolute pixel coordinates
        height, width = image.shape[:2]
        x_min = int((x - w/2) * width)
        y_min = int((y - h/2) * height)
        x_max = int((x + w/2) * width)
        y_max = int((y + h/2) * height)
        
        # Draw box
        cv2.rectangle(result, (x_min, y_min), (x_max, y_max), colors[i], 2)
        
        # Draw label if provided
        if labels is not None and i < len(labels):
            cv2.putText(result, labels[i], (x_min, y_min - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 2)
    
    return result


def save_image(image: np.ndarray, filepath: str) -> None:
    """
    Save an image to a file.
    
    Args:
        image: Image to save
        filepath: Path to save the image
    """
    cv2.imwrite(filepath, image)


def apply_augmentation(image: np.ndarray, augmentation_type: str, **kwargs) -> np.ndarray:
    """
    Apply augmentation to an image.
    
    Args:
        image: Image to augment
        augmentation_type: Type of augmentation ('rotate' or 'noise')
        **kwargs: Additional arguments for the augmentation
        
    Returns:
        np.ndarray: Augmented image
    """
    result = image.copy()
    
    if augmentation_type == 'rotate':
        angle = kwargs.get('angle', np.random.randint(-15, 15))
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        result = cv2.warpAffine(image, rotation_matrix, (width, height))
    
    elif augmentation_type == 'noise':
        noise_level = kwargs.get('noise_level', 25)
        noise = np.random.uniform(-noise_level, noise_level, image.shape).astype(np.float32)
        result = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    return result


def normalize_box(box: Tuple[int, int, int, int], image_size: Tuple[int, int]) -> List[float]:
    """
    Normalize a bounding box to YOLO format.
    
    Args:
        box: Bounding box (x_min, y_min, x_max, y_max)
        image_size: Image size (width, height)
        
    Returns:
        List[float]: Normalized box [x, y, w, h]
    """
    x_min, y_min, x_max, y_max = box
    width, height = image_size
    
    x = (x_min + x_max) / (2 * width)
    y = (y_min + y_max) / (2 * height)
    w = (x_max - x_min) / width
    h = (y_max - y_min) / height
    
    return [x, y, w, h]


def denormalize_box(box: List[float], image_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
    """
    Denormalize a YOLO format bounding box.
    
    Args:
        box: Normalized box [x, y, w, h]
        image_size: Image size (width, height)
        
    Returns:
        Tuple[int, int, int, int]: Bounding box (x_min, y_min, x_max, y_max)
    """
    x, y, w, h = box
    width, height = image_size
    
    x_min = int((x - w/2) * width)
    y_min = int((y - h/2) * height)
    x_max = int((x + w/2) * width)
    y_max = int((y + h/2) * height)
    
    return (x_min, y_min, x_max, y_max)
