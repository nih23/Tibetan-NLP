from tibetanDataGenerator.utils.data_loader import TextFactory
from tibetanDataGenerator.data.text_renderer import ImageBuilder
from tibetanDataGenerator.data.augmentation import RotateAugmentation
from tibetanDataGenerator.utils.bounding_box import BoundingBoxCalculator

#Parameter
p_bg = "data/background_images_train/Dalle_1.png"
p_font = "ext/Microsoft Himalaya.ttf"
box_pos = (50, 50)
box_size = (300, 150)

# Text generieren
text_generator = TextFactory.create_text_source("corpus", "data/corpora/Tibetan Number Words/")
text = text_generator.generate_text()

# Bild bauen
builder = ImageBuilder()

image = (
    builder.set_background(p_bg)
    .set_font(p_font, font_size=24)
    .add_text(text, (50, 50), (300, 150))
    .add_bounding_box((50, 50), (300, 150))
)

# Bounding Box berechnen
#bbox = BoundingBoxCalculator.calculate(text,  font=p_font)

# Debugging
#print(f"Bounding Box: {bbox}")
image.show()