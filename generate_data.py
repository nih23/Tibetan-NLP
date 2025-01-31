import argparse
import multiprocessing
import random
import re
import os
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

def generate_synthetic_image(images, label_dict, folder_with_background, folder_with_corpoare, folder_for_train_data, debug = False, font_path ='res/Microsoft Himalaya.ttf', single_label = False, image_size=1024, augmentation="noise"):
    ctr = hash_current_time()  # Assuming this function is defined elsewhere
    font_size = 24
    border_offset = int(0.05 * image_size)

    # Select random background image
    image_path = os.path.join(folder_with_background, random.choice(images))

    # Create ImageBuilder
    builder = ImageBuilder((image_size, image_size))
    builder.set_background(image_path)
    builder.set_font(font_path, font_size=font_size)

    # Generate text
    text_generator = TextFactory.create_text_source("corpus", folder_with_corpoare)
    text, file_name = text_generator.generate_text()

    # Calculate bounding box
    dx, dy = image_size, image_size

    bbox_str = ""

    max_box_size_w = random.randint(100, image_size)
    max_box_size = (max_box_size_w, 400)

    fitted_box_size = BoundingBoxCalculator.fit(text, max_box_size, font_size=font_size,
                                                font_path=font_path)

    box_pos_x = random.randint(border_offset, image_size - (fitted_box_size[0]+border_offset))
    box_pos_y = random.randint(border_offset, image_size - (fitted_box_size[1]+border_offset))
    box_position = (box_pos_x, box_pos_y)

    # Add text and bounding box
    builder.add_text(text, box_position, fitted_box_size)
    builder.add_bounding_box(box_position, fitted_box_size)

    # Apply augmentation
    augmentation = augmentation_strategies[augmentation.lower()]
    builder.apply_augmentation(augmentation)

    # Prepare bounding box string
    label = next(iter(label_dict.keys())) if single_label else os.path.splitext(file_name)[0]
    label_id = label_dict[label]
    x, y = box_position
    w, h = fitted_box_size
    bbox_str += f"{label_id} {x / dx} {y / dy} {w / dx} {h / dy}\n"

    # Save image
    image_filename = f"{label}_{ctr}.png"
    image_path = os.path.join(folder_for_train_data, 'images', image_filename)
    builder.save(image_path)

    # Ensure the labels directory exists
    labels_dir = os.path.join(folder_for_train_data, 'labels')
    os.makedirs(labels_dir, exist_ok=True)

    # Save label
    label_filename = f"{label}_{ctr}.txt"
    label_path = os.path.join(labels_dir, label_filename)
    with open(label_path, 'w') as f:
        f.write(bbox_str)

    if debug:
        print(f"Generated sample: {image_filename}")
        print(f"Bounding boxes:\n{bbox_str}")

    return image_filename, label_filename

def fill_label_dict(folder_path):
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


def generate_dataset(args, validation = False):
    folder_with_background = args.background_train
    folder_for_train_data = f'{args.dataset_name}/train/'
    no_samples = args.train_samples
    if(validation):
        folder_with_background = args.background_val
        folder_for_train_data = f'{args.dataset_name}/val/'
        no_samples = args.val_samples

    label_dict = {'tibetan': 0}
    if not args.single_label:
        label_dict = fill_label_dict(args.corpora_folder)

    # Load background images
    images = [file for file in os.listdir(folder_with_background) if file.lower().endswith(('.jpg', '.png'))]

    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

    args = (images, label_dict, folder_with_background, args.corpora_folder, folder_for_train_data, False, args.font_path, args.single_label, args.image_size, args.augmentation)

    number_of_calls = no_samples
    max_parallel_calls = os.cpu_count()

    # Create a pool of workers, limited to no. cpu parallel processes for generation of training data

    #results = generate_synthetic_image(*args)
    with multiprocessing.Pool(max_parallel_calls) as pool:
        results = pool.starmap(generate_synthetic_image, [args] * number_of_calls)

    label_dict_swap = {v: k for k, v in label_dict.items()} # swap key & value of dictionary for ultralytics yolo file format

    dataset_dict = OrderedDict([
        ('path', f"../{folder_for_train_data}"),
        ('train', 'train/images'),
        ('val', 'val/images'),
        ('nc', len(label_dict_swap)),  # Add the number of classes
        ('names', label_dict_swap)
    ])
    return dataset_dict


def main():
    parser = argparse.ArgumentParser(description="Generate YOLO dataset for Tibetan text detection")

    parser.add_argument('--background_train', type=str, default='./data/background_images_train/',
                        help='Folder with background images for training')
    parser.add_argument('--background_val', type=str, default='./data/background_images_val/',
                        help='Folder with background images for validation')
    parser.add_argument('--dataset_name', type=str, default='yolo_tibetan/',
                        help='Folder for the generated YOLO dataset')
    parser.add_argument('--corpora_folder', type=str, default='./data/corpora/Tibetan Number Words/',
                        help='Folder with Tibetan tibetan numbers corpora')
    parser.add_argument('--train_samples', type=int, default=100,
                        help='Number of training samples to generate')
    parser.add_argument('--val_samples', type=int, default=100,
                        help='Number of validation samples to generate')
    parser.add_argument('--no_cols', type=int, default=1,
                        help='Number of text columns to generate [1....5]')
    parser.add_argument('--font_path', type=str, default='ext/Microsoft Himalaya.ttf',
                        help='Path to a font file that supports Tibetan characters')
    parser.add_argument('--single_label', action='store_true',
                        help='Use a single label "tibetan" for all files instead of using filenames as labels')
    parser.add_argument('--image_size', type=int, default=1024,
                        help='size (pixels) of each image')
    parser.add_argument("--augmentation", choices=list(augmentation_strategies.keys()), default='noise',
                        help="Type of augmentation to apply")

    args = parser.parse_args()
    # Read default settings

    datasets_dir = Path(DATASETS_DIR)
    path = str(datasets_dir / args.dataset_name)
    args.dataset_name = path
    print(f"Generating YOLO dataset {args.dataset_name}...")

    dataset_dict = generate_dataset(args, validation = False)
    generate_dataset(args, validation = True)

    dataset_dict['path'] = args.dataset_name


    def represent_ordereddict(dumper, data):
        return dumper.represent_mapping('tag:yaml.org,2002:map', data.items())

    yaml.add_representer(OrderedDict, represent_ordereddict)

    with open(f"{args.dataset_name}/ultralytics.yml", 'w') as yaml_file:
        yaml.dump(dataset_dict, yaml_file, default_flow_style=False)

    print("Training can be started with the following command:\n\n"
          f"yolo detect train data=yolo detect train data={args.dataset_name}/ultralytics.yml epochs=100 imgsz=1024 model=yolo11n.pt\n"
          "")

if __name__ == "__main__":
    main()