#!/usr/bin/env python3
"""
Skript zur Generierung von Trainingsdaten für Tibetische OCR.
Erstellt synthetische Bilder mit Tibetischem Text für YOLO-Training.
"""

from pathlib import Path
from collections import OrderedDict
from ultralytics.data.utils import DATASETS_DIR
from tibetanDataGenerator.dataset_generator import generate_dataset

# Importiere Funktionen aus der tibetan_utils-Bibliothek
from tibetan_utils.arg_utils import create_generate_dataset_parser
from tibetan_utils.io_utils import save_yaml


def main():
    # Parse arguments
    parser = create_generate_dataset_parser()
    args = parser.parse_args()

    # Set dataset path
    datasets_dir = Path(DATASETS_DIR)
    path = str(datasets_dir / args.dataset_name)
    args.dataset_name = path
    print(f"Generiere YOLO-Datensatz {args.dataset_name}...")

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

    # Save dataset configuration
    yaml_path = f"{args.dataset_name}/data.yml"
    save_yaml(dataset_dict, yaml_path)

    print("Datensatzgenerierung abgeschlossen.")
    print("Training kann mit folgendem Befehl gestartet werden:\n")
    print(f"yolo detect train data={yaml_path} epochs=100 imgsz=1024 model=yolov8n.pt")


if __name__ == "__main__":
    main()
