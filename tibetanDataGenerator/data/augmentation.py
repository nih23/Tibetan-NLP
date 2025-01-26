import random


class AugmentationStrategy:
    def apply(self, image):
        raise NotImplementedError

class RotateAugmentation(AugmentationStrategy):
    def apply(self, image):
        return image.rotate(random.randint(-15, 15))

class NoiseAugmentation(AugmentationStrategy):
    def apply(self, image):
        # Füge Rauschen hinzu
        pass