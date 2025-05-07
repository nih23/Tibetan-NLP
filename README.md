# Tibetan OCR Tools

## Overview

This Python project focuses on detecting columns or text blocks of Tibetan texts in images. It provides a complete pipeline from dataset generation to inference:

1. **Dataset Generation**: Create synthetic training data by embedding Tibetan text into background images
2. **Model Training**: Train a YOLO-based object detection model with optional Weights & Biases logging
3. **Inference**: Detect Tibetan text blocks in new images, including support for Staatsbibliothek zu Berlin digital collections

![Validation results](res/results_val_1.png)

![Validation results](res/results_val_2.png)

## Quick Start Guide

### Installation

```bash
# Clone the repository
git clone https://github.com/nih23/Tibetan-NLP.git
cd Tibetan-NLP

# Install dependencies
pip install -r requirements.txt
```

### Complete Workflow

```bash
# 1. Generate dataset
python generate_dataset.py --train_samples 1000 --val_samples 200 --image_size 1024

# 2. Train model
python train_model.py --epochs 100 --export

# 3. Run inference
# On local images:
yolo predict task=detect model=runs/detect/train/weights/best.torchscript imgsz=1024 source=data/my_inference_data/*.jpg

# On Staatsbibliothek zu Berlin data:
python inference_sbb.py --ppn PPN12345678 --model runs/detect/train/weights/best.torchscript
```

## Features

- **Automated Dataset Generation**: Create training and validation datasets for Tibetan text detection
- **Customizable Parameters**: Configure background images, corpora, font, image size, and more
- **Text Generation Options**: Use synthetic Tibetan text or existing corpora
- **Advanced Image Processing**: Apply rotation and noise augmentation strategies
- **Multiprocessing Support**: Leverage parallel processing for efficient dataset generation
- **Experiment Tracking**: Integration with Weights & Biases for monitoring training progress
- **Specialized Inference**: Support for Staatsbibliothek zu Berlin digital collections

## Detailed Documentation

### 1. Dataset Generation

The dataset generation script creates synthetic training data by embedding Tibetan text into background images.

```bash
python generate_dataset.py --train_samples 1000 --val_samples 200 --augmentation rotate
```

#### Dataset Generation Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--background_train` | Folder with background images for training | './data/background_images_train/' |
| `--background_val` | Folder with background images for validation | './data/background_images_val/' |
| `--dataset_name` | Folder for the generated YOLO dataset | 'yolo_tibetan/' |
| `--corpora_folder` | Folder with Tibetan corpora | './data/corpora/Tibetan Number Words/' |
| `--train_samples` | Number of training samples to generate | 100 |
| `--val_samples` | Number of validation samples to generate | 100 |
| `--no_cols` | Number of text columns to generate [1-5] | 1 |
| `--font_path` | Path to a Tibetan font file | 'ext/Microsoft Himalaya.ttf' |
| `--single_label` | Use a single label "tibetan" for all files | flag |
| `--debug` | Enable debug mode for verbose output | flag |
| `--image_size` | Size of generated images in pixels | 1024 |
| `--augmentation` | Type of augmentation to apply ['rotate', 'noise'] | 'noise' |

### 2. Model Training

After generating the dataset, you can train a YOLO model using our dedicated training script.

```bash
python train_model.py --epochs 100 --imgsz 1024 --export
```

#### Training Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--dataset` | Name of the dataset folder | 'yolo_tibetan/' |
| `--model` | Path to the base model | 'yolov8n.pt' |
| `--epochs` | Number of training epochs | 100 |
| `--batch` | Batch size for training | 16 |
| `--imgsz` | Image size for training | 1024 |
| `--workers` | Number of workers for data loading | 8 |
| `--device` | Device for training (e.g., cpu, 0, 0,1,2,3) | '' |
| `--project` | Project name for output | 'runs/detect' |
| `--name` | Experiment name | 'train' |
| `--export` | Export the model after training as TorchScript | flag |
| `--patience` | EarlyStopping patience in epochs | 50 |

#### Weights & Biases Integration

The training script includes integration with [Weights & Biases](https://wandb.ai/) for experiment tracking and visualization.

```bash
python train_model.py --epochs 100 --wandb --wandb-project TibetanOCR
```

| Argument | Description | Default |
|----------|-------------|---------|
| `--wandb` | Enable Weights & Biases logging | flag |
| `--wandb-project` | W&B project name | 'TibetanOCR' |
| `--wandb-entity` | W&B entity (team or username) | None |
| `--wandb-tags` | Comma-separated tags for the experiment | None |
| `--wandb-name` | Name of the experiment in wandb | same as --name |

When wandb logging is enabled, the script will:
- Log training metrics (loss, mAP, precision, recall, etc.)
- Upload dataset samples for visualization
- Save model checkpoints as artifacts
- Generate plots and confusion matrices

#### Alternative: Ultralytics CLI

You can also use the Ultralytics CLI directly:

```bash
# Train the model
yolo detect train data=yolo_tibetan/data.yml epochs=100 imgsz=1024 model=yolov8n.pt

# Export the model
yolo detect export model=runs/detect/train/weights/best.pt 
```

### 3. Inference

#### Standard Inference

For inference on local image files:

```bash
yolo predict task=detect model=runs/detect/train/weights/best.torchscript imgsz=1024 source=data/my_inference_data/*.jpg
```

The results are saved to folder `runs/detect/predict`

#### Inference on Staatsbibliothek zu Berlin Data

For inference on documents from the Staatsbibliothek zu Berlin, use our specialized script:

```bash
python inference_sbb.py --ppn PPN12345678 --model runs/detect/train/weights/best.torchscript
```

If you encounter SSL certificate issues, you can use the `--no-ssl-verify` option:

```bash
python inference_sbb.py --ppn PPN12345678 --model runs/detect/train/weights/best.torchscript --no-ssl-verify
```

| Argument | Description | Default |
|----------|-------------|---------|
| `--ppn` | PPN (Pica Production Number) of the document | required |
| `--model` | Path to the trained model | required |
| `--imgsz` | Image size for inference | 1024 |
| `--conf` | Confidence threshold for detections | 0.25 |
| `--download` | Download images instead of processing them directly | flag |
| `--output` | Directory for saving downloaded images | 'sbb_images' |
| `--max-images` | Maximum number of images for inference (0 = all) | 0 |
| `--no-ssl-verify` | Disable SSL certificate verification | flag |

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
