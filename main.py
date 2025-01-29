from tibetanDataGenerator.utils.data_loader import TextFactory
from tibetanDataGenerator.data.text_renderer import ImageBuilder
from tibetanDataGenerator.data.augmentation import RotateAugmentation, NoiseAugmentation
from tibetanDataGenerator.utils.bounding_box import BoundingBoxCalculator

#Parameter
p_bg = "data/background_images_train/Dalle_1.png"
p_font = "ext/Microsoft Himalaya.ttf"
image_size = (768,768)
box_pos = (200, 200)
box_size = (300, 150)
font_size = 45

# Text generieren
text_generator = TextFactory.create_text_source("synthetic")
#text_generator = TextFactory.create_text_source("corpus", "data/corpora/Tibetan Number Words/")
text = text_generator.generate_text()

# Bild bauen
builder = ImageBuilder(image_size)

# Bounding Box berechnen
fitted_box_size = BoundingBoxCalculator.fit(text, box_size, font_size=font_size, font_path=p_font)

image = (
    builder.set_background(p_bg)
    .set_font(p_font, font_size=font_size)
    .add_text(text, box_pos, fitted_box_size)
    .apply_augmentation(NoiseAugmentation())
    .add_bounding_box(box_pos, fitted_box_size)
)

# Debugging
image.show()