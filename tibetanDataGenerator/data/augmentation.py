import random

import numpy as np
from PIL import Image

class AugmentationStrategy:
    def apply(self, image: Image):
        raise NotImplementedError

class RotateAugmentation(AugmentationStrategy):
    def apply(self, image: Image):
        return image.rotate(random.randint(-15, 15))

class NoiseAugmentation(AugmentationStrategy):
    def apply(self, image: Image):
        # Convert PIL Image to numpy array
        img_array = np.array(image).astype(np.float32)

        # Add noise
        noise = np.random.uniform(-25.5, 25.5, img_array.shape)
        noisy_img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)

        # Convert back to PIL Image
        return Image.fromarray(noisy_img_array)