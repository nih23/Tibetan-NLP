# Tibetan Column Detection

## Overview
This Python project focuses on generating training data for detecting columns or text blocks of tibetan texts by embedding Tibetan text into images. It includes functions to create lorem ipsum-like Tibetan text, read random Tibetan text files from a directory, and calculate and embed text within specified bounding boxes in images. The project effectively handles Tibetan script, ensuring proper display and formatting within the images.

## Features
- **Tibetan Text Generation**: Generate lorem ipsum-like Tibetan text.
- **Read Random Tibetan Text Files**: Randomly select and read Tibetan text files from a specified directory.
- **Text Embedding in Images**: Embed Tibetan text in images with options for wrapping and bounding box size calculations.

## Installation

To use this project, clone the repository:

```bash
git clone https://github.com/nih23/tibetan-column-detection.git
cd tibetan-text-image-embedder
```

## Usage

### Generating Tibetan Text

```python
from generate_training_data import generate_lorem_like_tibetan_text

tibetan_text = generate_lorem_like_tibetan_text(100)
print(tibetan_text)
```

### Reading Random Tibetan Text File

```python
from generate_training_data import read_random_tibetan_file

directory_path = '/path/to/tibetan/text/files'
tibetan_text = read_random_tibetan_file(directory_path)
print(tibetan_text)
```

### Embedding Text in Image

```python
from generate_training_data import embed_text_in_box_with_limit
from PIL import Image

# Load your image
image_path = '/path/to/your/image.jpg'
image = Image.open(image_path)

# Specify text and box details
text_to_embed = "Your Tibetan Text Here"
box_position = (50, 50)
box_size = (400, 100)

# Embed the text
embedded_image = embed_text_in_box_with_limit(image, text_to_embed, box_position, box_size)
embedded_image.show()
```

### Generating training data
```bash
python generate_training_data.py
```

### Train YOLOv8n
```bash
yolo detect train data=data/yolo_tibetan/tibetan_text_boxes.yml epochs=100 imgsz=1024
```

Training of YOLOv8n is done by a CLI call to [Ultralytics](https://docs.ultralytics.com/usage/cli/#train).

## Contributions

Contributions to this project are welcome! Please fork the repository and submit a pull request with your proposed changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
