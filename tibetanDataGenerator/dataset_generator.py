import argparse
import multiprocessing
import random
import re
import os
from typing import Tuple, Dict, List

import yaml
from pathlib import Path
from collections import OrderedDict
from ultralytics.data.utils import DATASETS_DIR
from tibetanDataGenerator.utils.data_loader import TextFactory
from tibetanDataGenerator.data.text_renderer import ImageBuilder
from tibetanDataGenerator.data.augmentation import RotateAugmentation, NoiseAugmentation
from tibetanDataGenerator.utils.bounding_box import BoundingBoxCalculator
from tibetanDataGenerator.utils.identifier import hash_current_time

# Define a dictionary of augmentation strategies
augmentation_strategies = {
    'rotate': RotateAugmentation(),
    'noise': NoiseAugmentation()
}


def generate_dataset(args: argparse.Namespace, validation: bool = False) -> Dict:
    """
    Generate a dataset for training or validation.

    Args:
        args (argparse.Namespace): Command-line arguments.
        validation (bool): Whether to generate validation dataset. Defaults to False.

    Returns:
        Dict: A dictionary containing dataset information.
    """
    dataset_info = _setup_dataset_info(args, validation)
    label_dict = _create_label_dict(args)
    background_images = _load_background_images(dataset_info['background_folder'])

    generation_args = _prepare_generation_args(args, dataset_info, label_dict, background_images)

    results = _generate_images_in_parallel(generation_args, dataset_info['no_samples'])

    return _create_dataset_dict(dataset_info['folder'], label_dict)


def generate_synthetic_image(
        images: List[str],
        label_dict: Dict[str, int],
        folder_with_background: str,
        folder_with_corpora: str,
        folder_for_train_data: str,
        debug: bool = True,
        font_path: str = 'res/Microsoft Himalaya.ttf',
        single_label: bool = False,
        image_size: int = 1024,
        augmentation: str = "noise"
) -> Tuple[str, str]:
    # Constants
    FONT_SIZE = 24
    BORDER_OFFSET_RATIO = 0.05

    ctr = hash_current_time()
    border_offset = int(BORDER_OFFSET_RATIO * image_size)

    # Image setup
    image_path = _select_random_background(folder_with_background, images)
    builder = _setup_image_builder(image_path, image_size, font_path, FONT_SIZE)

    # Text generation and positioning
    text, file_name = _generate_text(folder_with_corpora)
    text_position, box_position, fitted_box_size = _calculate_text_position(
        text, image_size, border_offset, font_path, FONT_SIZE
    )

    # Add text and bounding box
    builder.add_text(text, text_position, fitted_box_size)
    if debug == True:
        builder.add_bounding_box(box_position, fitted_box_size)

    # Apply augmentation
    _apply_augmentation(builder, augmentation)

    # Prepare and save image and label
    image_filename, label_filename = _save_image_and_label(
        builder, text, ctr, folder_for_train_data, label_dict,
        single_label, file_name, box_position, fitted_box_size,
        image_size, debug
    )

    return image_filename, label_filename


def _select_random_background(folder: str, images: List[str]) -> str:
    return os.path.join(folder, random.choice(images))


def _setup_image_builder(image_path: str, image_size: int, font_path: str, font_size: int) -> ImageBuilder:
    builder = ImageBuilder((image_size, image_size))
    builder.set_background(image_path)
    builder.set_font(font_path, font_size=font_size)
    return builder


def _generate_text(folder_with_corpora: str) -> Tuple[str, str]:
    text_generator = TextFactory.create_text_source("corpus", folder_with_corpora)
    return text_generator.generate_text()


def _calculate_text_position(
        text: str,
        image_size: int,
        border_offset: int,
        font_path: str,
        font_size: int
) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    max_box_size_w = random.randint(100, image_size)
    max_box_size = (max_box_size_w, 400)

    fitted_box_size = BoundingBoxCalculator.fit(text, max_box_size, font_size=font_size, font_path=font_path)

    text_pos_x = random.randint(border_offset, image_size - (fitted_box_size[0] + border_offset))
    text_pos_y = random.randint(border_offset, image_size - (fitted_box_size[1] + border_offset))

    text_position = (text_pos_x, text_pos_y)
    box_position = (text_pos_x + int(fitted_box_size[0] / 2), text_pos_y - int(fitted_box_size[1] / 2))

    return text_position, box_position, fitted_box_size


def _apply_augmentation(builder: ImageBuilder, augmentation: str):
    augmentation_strategy = augmentation_strategies[augmentation.lower()]
    builder.apply_augmentation(augmentation_strategy)


def _save_image_and_label(
        builder: ImageBuilder,
        text: str,
        ctr: str,
        folder_for_train_data: str,
        label_dict: Dict[str, int],
        single_label: bool,
        file_name: str,
        box_position: Tuple[int, int],
        fitted_box_size: Tuple[int, int],
        image_size: int,
        debug: bool
) -> Tuple[str, str]:
    label = next(iter(label_dict.keys())) if single_label else os.path.splitext(file_name)[0]
    label_id = label_dict[label]

    image_filename = f"{label}_{ctr}.png"
    image_path = os.path.join(folder_for_train_data, 'images', image_filename)
    builder.save(image_path)

    bbox_str = _create_bbox_string(label_id, box_position, fitted_box_size, image_size)

    labels_dir = os.path.join(folder_for_train_data, 'labels')
    os.makedirs(labels_dir, exist_ok=True)

    label_filename = f"{label}_{ctr}.txt"
    label_path = os.path.join(labels_dir, label_filename)
    with open(label_path, 'w') as f:
        f.write(bbox_str)

    if debug == True:
        print(f"Generated sample: {image_filename}")
        print(f"Bounding boxes:\n{bbox_str}")

    return image_filename, label_filename


def _create_bbox_string(label_id: int, box_position: Tuple[int, int], box_size: Tuple[int, int], image_size: int) -> str:
    x, y = box_position
    w, h = box_size
    return f"{label_id} {x / image_size} {y / image_size} {w / image_size} {h / image_size}\n"


def _fill_label_dict(folder_path):
    label_dict = {}
    label_id = 0

    # Get all txt files
    files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]

    # Sort files based on the numeric part
    sorted_files = sorted(files, key=lambda x: int(re.findall(r'\d+', x)[-1]))

    for filename in sorted_files:
        label = os.path.splitext(filename)[0]
        if label not in label_dict:
            label_dict[label] = label_id
            label_id += 1

    return label_dict


def _setup_dataset_info(args: argparse.Namespace, validation: bool) -> Dict:
    """Set up basic dataset information based on validation flag."""
    if validation:
        return {
            'background_folder': args.background_val,
            'folder': f'{args.dataset_name}/val/',
            'no_samples': args.val_samples
        }
    else:
        return {
            'background_folder': args.background_train,
            'folder': f'{args.dataset_name}/train/',
            'no_samples': args.train_samples
        }


def _create_label_dict(args: argparse.Namespace) -> Dict[str, int]:
    """Create a dictionary of labels based on single_label flag."""
    if args.single_label:
        return {'tibetan': 0}
    else:
        return _fill_label_dict(args.corpora_folder)


def _load_background_images(folder: str) -> List[str]:
    """Load background image filenames from the specified folder."""
    return [file for file in os.listdir(folder) if file.lower().endswith(('.jpg', '.png'))]


def _prepare_generation_args(args: argparse.Namespace, dataset_info: Dict, label_dict: Dict,
                             images: List[str]) -> Tuple:
    """Prepare arguments for image generation."""
    return (
        images, label_dict, dataset_info['background_folder'], args.corpora_folder,
        dataset_info['folder'], args.debug, args.font_path, args.single_label,
        args.image_size, args.augmentation
    )


def _generate_images_in_parallel(generation_args: Tuple, no_samples: int) -> List:
    """Generate images in parallel using multiprocessing."""
    max_parallel_calls = os.cpu_count()
    with multiprocessing.Pool(max_parallel_calls) as pool:
        return pool.starmap(generate_synthetic_image, [generation_args] * no_samples)


def _create_dataset_dict(folder: str, label_dict: Dict[str, int]) -> OrderedDict:
    """Create a dictionary containing dataset information."""
    label_dict_swap = {v: k for k, v in label_dict.items()}
    return OrderedDict([
        ('path', f"../{folder}"),
        ('train', 'train/images'),
        ('val', 'val/images'),
        ('nc', len(label_dict_swap)),
        ('names', label_dict_swap)
    ])