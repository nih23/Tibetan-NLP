import argparse
from pathlib import Path
import yaml
from collections import OrderedDict
from ultralytics.data.utils import DATASETS_DIR
from tibetanDataGenerator.dataset_generator import generate_dataset


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
    parser.add_argument('--debug', action='store_true',
                        help='More verbose output with debug information about the image generation process.')
    parser.add_argument('--image_size', type=int, default=1024,
                        help='size (pixels) of each image')
    parser.add_argument("--augmentation", choices=['rotate', 'noise'], default='noise',
                        help="Type of augmentation to apply")

    args = parser.parse_args()

    datasets_dir = Path(DATASETS_DIR)
    path = str(datasets_dir / args.dataset_name)
    args.dataset_name = path
    print(f"Generating YOLO dataset {args.dataset_name}...")

    # Generate training dataset
    train_dataset_dict = generate_dataset(args, validation=False)

    # Generate validation dataset
    val_dataset_dict = generate_dataset(args, validation=True)

    # Combine train and val dataset information
    dataset_dict = {
        'path': args.dataset_name,
        'train': 'train/images',
        'val': 'val/images',
        'nc': train_dataset_dict['nc'],
        'names': train_dataset_dict['names']
    }

    def represent_ordereddict(dumper, data):
        return dumper.represent_mapping('tag:yaml.org,2002:map', data.items())

    yaml.add_representer(OrderedDict, represent_ordereddict)

    with open(f"{args.dataset_name}/data.yml", 'w') as yaml_file:
        yaml.dump(dataset_dict, yaml_file, default_flow_style=False)

    print("Dataset generation completed.")
    print("Training can be started with the following command:\n")
    print(f"yolo detect train data={args.dataset_name}/data.yml epochs=100 imgsz=1024 model=yolov8n.pt")


if __name__ == "__main__":
    main()