from PIL import Image, ImageDraw, ImageFont
from yolo_utils import prepare_bbox_string

import io
import os
import random
import yaml
import numpy as np
from tqdm import tqdm
from utils import hash_current_time

import multiprocessing

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


def generate_sample(images, label, label_id, folder_with_background, folder_with_corpoare, folder_for_train_data, debug = False):
    ctr = hash_current_time()

    image_id = random.randint(0, len(images)-1)
    image_path = folder_with_background + "/" + images[image_id]
    with open(image_path, "rb") as image_file:
        magazine_image = Image.open(io.BytesIO(image_file.read()))

    dx, dy = magazine_image.size
    no_cols = random.randint(1, 3)
    dx_multicol = int(dx / no_cols)

    max_box_size_w = random.randint(100, dx_multicol-5)
    max_box_size = (max_box_size_w, 400)    # maximum size of text box (text will be wrapped if longer)
    box_pos_x = random.randint(0, dx_multicol - max_box_size[0])
    box_pos_y = random.randint(0, dy - max_box_size[1])

    bbox_str = ""

    if(debug):
        print(f"\n\n[{ctr}] image size: ({dx},{dy})")

    for i in range(no_cols):
        #text_to_embed = generate_lorem_like_tibetan_text(500)
        text_to_embed = read_random_tibetan_file(folder_with_corpoare)

        # position of bounding box
        if(i>0): # shift to the right for columns 2+
            box_pos_x += max_box_size_w + i*random.randint(5, 30)
        box_position = (box_pos_x, box_pos_y)  # position of text box
        
        if(debug):
            print(f"position of box in col {i}: ({box_position[0]},{box_position[1]})")
            print(f" >> max size ({max_box_size[0]},{max_box_size[1]})")

        bbox = calculate_wrapped_text_bounding_box(text_to_embed, max_box_size)
        magazine_image = embed_text_in_box_with_limit(magazine_image, text_to_embed, box_position, max_box_size)
        bbox = np.array(bbox)
        x = box_position[0]
        y = box_position[1]
        w = bbox[0]
        h = bbox[1]

        bbox_str += prepare_bbox_string(label_id,x,y,h,w,dx,dy) + "\n"

    pImg = folder_for_train_data + "/images/" + label + "_" + str(ctr) + ".png"
    pBB = folder_for_train_data + "/labels/" + label + "_" + str(ctr) + ".txt" 

    ensure_directory_exists(pImg)
    ensure_directory_exists(pBB)

    # Save the image
    magazine_image.save(pImg)

    # Open the file in write mode
    with open(pBB, 'w') as file:
        # Write the string to the file
        file.write(bbox_str)


def generate_data(no_images = 5000, folder_with_background = './data/background_images/', folder_for_train_data = './data/train/', folder_with_corpoare = 'data/corpora/UVA Tibetan Spoken Corpus/'):

    label = 'tibetan'
    label_id = 0
    label_dict = {}
    label_dict[label] = label_id

    # Load background images
    images = [file for file in os.listdir(folder_with_background) if file.lower().endswith(('.jpg', '.png'))]

    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

    args = (images, label, label_id, folder_with_background, folder_with_corpoare, folder_for_train_data)

    number_of_calls = no_images
    max_parallel_calls = os.cpu_count()

    # Create a pool of workers, limited to no. cpu parallel processes for generation of training data
    with multiprocessing.Pool(max_parallel_calls) as pool:
        results = pool.starmap(generate_sample, [args] * number_of_calls)

    label_dict_swap = {v: k for k, v in label_dict.items()} # swap key & value of dictionary for ultralytics yolo file format
    dataset_dict = {'path': f"../{folder_for_train_data}", 'train': 'train/images', 'val': 'val/images', 'names': label_dict_swap }

    return dataset_dict


if __name__ == "__main__":
    folder_with_background_train = './data/background_images_train/'
    folder_with_background_val = './data/background_images_val/'
    folder_for_dataset = './data/yolo_tibetan'
    folder_for_train_data = f'{folder_for_dataset}/train/'
    folder_for_val_data = f'{folder_for_dataset}/val/'
    folder_with_corpoare = 'data/corpora/UVA Tibetan Spoken Corpus/'

    dataset_dict = generate_data(5000, folder_with_background_train, folder_for_train_data, folder_with_corpoare)
    generate_data(500, folder_with_background_val, folder_for_val_data, folder_with_corpoare)
    dataset_dict['path'] = folder_for_dataset

    with open(f"{folder_for_dataset}/tibetan_text_boxes.yml", 'w') as yaml_file:
        yaml.dump(dataset_dict, yaml_file, default_flow_style=False)
