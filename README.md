# Tibetan OCR Tools

## Overview
This Python project focuses on generating training data for detecting columns or text blocks of Tibetan texts by embedding Tibetan text into images. It provides a flexible and customizable way to create datasets for machine learning models, particularly for YOLO-based object detection tasks.

![Validation results](res/results_val_1.png)

![Validation results](res/results_val_2.png)

The project includes functionality for generating synthetic Tibetan text, reading from existing Tibetan corpora, applying various augmentations, and creating YOLO-compatible datasets.

## Features
- **Automated Dataset Generation**: Simplifies the process of creating training and validation datasets for Tibetan text detection.
- **Customizable Input**: Allows users to specify various parameters like background images, corpora, font, image size, and more.
- **Text Generation**: Supports both synthetic Tibetan text generation and reading from existing corpora.
- **Image Processing**: Utilizes PIL for image manipulation and text rendering.
- **Augmentation**: Includes rotation and noise augmentation strategies.
- **Multiprocessing Support**: Leverages parallel processing for efficient dataset generation.
- **YOLO Compatibility**: Generates datasets in a format compatible with YOLO training.

## Getting Started

### Prerequisites
- Python 3.x
- PIL (Python Imaging Library)
- PyYAML
- Ultralytics YOLO
- Additional Python libraries: numpy, tqdm, etc.

### Installation
Clone the repository to your local machine:

```bash
git clone https://github.com/nih23/Tibetan-NLP.git
cd Tibetan-NLP
```

Install required packages:
```bash
pip install -r requirements.txt
```

### Usage
Run the dataset generation script with desired arguments:
```bash
python generate_dataset.py --train_samples 1000 --val_samples 200 --augmentation rotate
```

### Command-line Arguments

- `--background_train`: Folder with background images for training (default: './data/background_images_train/')
- `--background_val`: Folder with background images for validation (default: './data/background_images_val/')
- `--dataset_name`: Folder for the generated YOLO dataset (default: 'yolo_tibetan/')
- `--corpora_folder`: Folder with Tibetan corpora (default: './data/corpora/Tibetan Number Words/')
- `--train_samples`: Number of training samples to generate (default: 100)
- `--val_samples`: Number of validation samples to generate (default: 100)
- `--no_cols`: Number of text columns to generate [1-5] (default: 1)
- `--font_path`: Path to a Tibetan font file (default: 'ext/Microsoft Himalaya.ttf')
- `--single_label`: Use a single label "tibetan" for all files (flag)
- `--debug`: Enable debug mode for verbose output (flag)
- `--image_size`: Size of generated images in pixels (default: 1024)
- `--augmentation`: Type of augmentation to apply ['rotate', 'noise'] (default: 'noise')


### Training with YOLO
After generating the dataset, you can train a YOLO model using our dedicated training script:

```bash
python train_model.py --epochs 100 --imgsz 1024 --export
```

#### Training Script Arguments

- `--dataset`: Name of the dataset folder (default: 'yolo_tibetan/')
- `--model`: Path to the base model (default: 'yolov8n.pt')
- `--epochs`: Number of training epochs (default: 100)
- `--batch`: Batch size for training (default: 16)
- `--imgsz`: Image size for training (default: 1024)
- `--workers`: Number of workers for data loading (default: 8)
- `--device`: Device for training (e.g., cpu, 0, 0,1,2,3 for multiple GPUs)
- `--project`: Project name for output (default: 'runs/detect')
- `--name`: Experiment name (default: 'train')
- `--export`: Export the model after training as TorchScript (flag)
- `--patience`: EarlyStopping patience in epochs (default: 50)

Alternatively, you can still use the Ultralytics CLI directly:

```bash
yolo detect train data=yolo_tibetan/data.yml epochs=100 imgsz=1024 model=yolov8n.pt
```

The model is then converted into a TorchScript for inference:
```bash
yolo detect export model=runs/detect/train/weights/best.pt 
```

### Inference
We can employ our trained model for recognition and classification of Tibetan text blocks in several ways:

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

##### SBB Inference Script Arguments

- `--ppn`: PPN (Pica Production Number) of the document in the Staatsbibliothek zu Berlin (required)
- `--model`: Path to the trained model (required)
- `--imgsz`: Image size for inference (default: 1024)
- `--conf`: Confidence threshold for detections (default: 0.25)
- `--download`: Download images instead of processing them directly (flag)
- `--output`: Directory for saving downloaded images (default: 'sbb_images')
- `--max-images`: Maximum number of images for inference (0 = all)

### Complete Workflow

Here's the complete start-to-end workflow:

1. **Generate Dataset**:
   ```bash
   python generate_dataset.py --train_samples 1000 --val_samples 200 --image_size 1024
   ```

2. **Train Model**:
   ```bash
   python train_model.py --epochs 100 --export
   ```

3. **Run Inference**:
   - On local images:
     ```bash
     yolo predict task=detect model=runs/detect/train/weights/best.torchscript imgsz=1024 source=data/my_inference_data/*.jpg
     ```
   - On Staatsbibliothek zu Berlin data:
     ```bash
     python inference_sbb.py --ppn PPN12345678 --model runs/detect/train/weights/best.torchscript
     ```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
