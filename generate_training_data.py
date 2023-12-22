from PIL import Image, ImageDraw, ImageFont
from yolo_utils import prepare_bbox_string

import io
import os
import random
import yaml
import numpy as np


def ensure_directory_exists(file_path):
    # Extract the directory part of the file_path
    directory = os.path.dirname(file_path)

    # Check if the directory exists
    if not os.path.exists(directory):
        # If the directory does not exist, create it
        os.makedirs(directory, exist_ok=True)


def calculate_wrapped_text_bounding_box(text, box_size, font_path='res/Microsoft Himalaya.ttf', font_size=24):
    """
    Calculate the true bounding box size for the specified text when it is wrapped and terminated to fit a given box size.

    :param text: Text to be measured.
    :param box_size: Tuple (width, height) specifying the size of the box to fit the text.
    :param font_path: Path to the font file.
    :param font_size: Size of the font.
    :return: Tuple (width, height) representing the actual bounding box size of the wrapped and terminated text.
    """
    # Create a dummy image to get a drawing context
    dummy_image = Image.new('RGB', (1, 1))
    draw = ImageDraw.Draw(dummy_image)

    # Define the font
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        font = ImageFont.load_default()
        print("Warning: Default font used, may not accurately measure text.")

    box_w, box_h = box_size
    actual_text_width, actual_text_height = 0, 0
    y_offset = 0

    # Process each line
    for line in text.split('\n'):
        while line:
            # Find the breakpoint for wrapping
            for i in range(len(line)):
                if draw.textlength(line[:i + 1], font=font) > box_w:
                    break
            else:
                i = len(line)

            # Add the line to wrapped text
            wrapped_line = line[:i]

            left, top, right, bottom = font.getbbox(wrapped_line)
            line_width, line_height = right - left, bottom - top

            actual_text_width = max(actual_text_width, line_width)
            y_offset += line_height

            # Check if the next line exceeds the box height
            if y_offset > box_h:
                y_offset -= line_height  # Remove the last line's height if it exceeds
                break

            line = line[i:]

        if y_offset > box_h:
            break

    return actual_text_width, y_offset


def read_random_tibetan_file(directory):
    """
    Read a random text file containing Tibetan text from a specified directory.

    :param directory: The directory containing Tibetan text files.
    :return: Content of a randomly selected text file.
    """
    # List all files in the specified directory
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    if not files:
        return "No files found in the specified directory."

    # Randomly select a file
    random_file = random.choice(files)
    file_path = os.path.join(directory, random_file)

    # Read the content of the file
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
    except Exception as e:
        return f"Error reading file {random_file}: {e}"

    return content



def generate_lorem_like_tibetan_text(length):
    """
    Generate a lorem ipsum like Tibetan text string of a specified length.
    
    This function creates words of random lengths and separates them with a space,
    similar to the structure of lorem ipsum text.
    """
    tibetan_range = (0x0F40, 0x0FBC)  # Restricting range to more common characters
    word_lengths = [random.randint(2, 10) for _ in range(length // 5)]
    
    words = []
    for word_length in word_lengths:
        word = ''.join(chr(random.randint(*tibetan_range)) for _ in range(word_length))
        words.append(word)

    return ' '.join(words)


def embed_text_in_box_with_limit(image, text, box_position, box_size, font_path='res/Microsoft Himalaya.ttf', font_size=24):
    """
    Embed text within a specified rectangular box on an image, terminating the text if it surpasses the bounding box.

    :param image: PIL Image object to embed text on.
    :param text: Text to be embedded.
    :param box_position: Tuple (x, y) specifying the top left corner of the box.
    :param box_size: Tuple (width, height) specifying the size of the box.
    :param font_path: Path to the font file.
    :param font_size: Size of the font.
    :return: Image object with text embedded within the box.
    """
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        font = ImageFont.load_default()
        print("Warning: Default font used, may not display text as expected.")

    box_x, box_y = box_position
    box_w, box_h = box_size
    max_y = box_y + box_h
    wrapped_text = []
    for line in text.split('\n'):
        while line:
            for i in range(len(line)):
                if draw.textlength(line[:i + 1], font=font) > box_w:
                    break
            else:
                i = len(line)

            wrapped_text.append(line[:i])
            line = line[i:]

    y_offset = 0
    for line in wrapped_text:
        left, top, right, bottom = font.getbbox(line)
        line_height = bottom - top
        if box_y + y_offset + line_height > max_y:
            break  # Stop if the next line exceeds the box height
        draw.text((box_x, box_y + y_offset), line, font=font, fill=(0, 0, 0))
        y_offset += line_height

    return image

def generate_data(no_images = 10, folder_with_background = './data/background_images/', folder_for_train_data = './data/train/', folder_with_corpoare = 'data/corpora/UVA Tibetan Spoken Corpus/'):

    label = 'tibetan'
    label_id = 0
    label_dict = {}
    label_dict[label] = label_id

    # Load background images
    images = [file for file in os.listdir(folder_with_background) if file.lower().endswith(('.jpg', '.png'))]

    for ctr in range(no_images):

        image_id = random.randint(0, len(images)-1)
        image_path = folder_with_background + "/" + images[image_id]
        with open(image_path, "rb") as image_file:
            magazine_image = Image.open(io.BytesIO(image_file.read()))

        dx, dy = magazine_image.size
        #text_to_embed = generate_lorem_like_tibetan_text(500)
        text_to_embed = read_random_tibetan_file(folder_with_corpoare)
        max_box_size = (400, 400)    # maximum size of text box (text will be wrapped if longer)

        box_pos_x = random.randint(0, dx - max_box_size[0])
        box_pos_y = random.randint(0, dy - max_box_size[1])
        box_position = (box_pos_x, box_pos_y)  # position of text box


        bbox = calculate_wrapped_text_bounding_box(text_to_embed, max_box_size)
        img = embed_text_in_box_with_limit(magazine_image, text_to_embed, box_position, max_box_size)
        bbox = np.array(bbox)
        x = box_position[0]
        y = box_position[1]
        w = bbox[0]
        h = bbox[1]

        bbox_str = prepare_bbox_string(label_id,x,y,h,w,dx,dy)

        pImg = folder_for_train_data + "/images/" + label + "_" + str(ctr) + ".png"
        pBB = folder_for_train_data + "/labels/" + label + "_" + str(ctr) + ".txt" 

        ensure_directory_exists(pImg)
        ensure_directory_exists(pBB)

        # Save the image
        img.save(pImg)

        # Open the file in write mode
        with open(pBB, 'w') as file:
            # Write the string to the file
            file.write(bbox_str)


    label_dict_swap = {v: k for k, v in label_dict.items()} # swap key & value of dictionary for yolo file format
    dataset_dict = {'path': f"../{folder_for_train_data}", 'train': 'train/images', 'val': 'val/images', 'names': label_dict_swap }

    return dataset_dict


if __name__ == "__main__":
    folder_with_background = './data/background_images/'
    folder_for_dataset = './data/yolo_tibetan'
    folder_for_train_data = f'{folder_for_dataset}/train/'
    folder_for_val_data = f'{folder_for_dataset}/val/'
    folder_with_corpoare = 'data/corpora/UVA Tibetan Spoken Corpus/'

    dataset_dict = generate_data(1000, folder_with_background, folder_for_train_data, folder_with_corpoare)
    generate_data(100, folder_with_background, folder_for_val_data, folder_with_corpoare)

    with open(f"{folder_for_dataset}/dataset.yml", 'w') as yaml_file:
        yaml.dump(dataset_dict, yaml_file, default_flow_style=False)
