import argparse
from tibetanDataGenerator.utils.data_loader import TextFactory
from tibetanDataGenerator.data.text_renderer import ImageBuilder
from tibetanDataGenerator.data.augmentation import RotateAugmentation, NoiseAugmentation
from tibetanDataGenerator.utils.bounding_box import BoundingBoxCalculator

# Define a dictionary of augmentation strategies
augmentation_strategies = {
    'rotate': RotateAugmentation(),
    'noise': NoiseAugmentation()
}

text_sources = {
    "synthetic": lambda _: TextFactory.create_text_source("synthetic"),
    "corpus": lambda path: TextFactory.create_text_source("corpus", path)
}

def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic Tibetan text images")
    parser.add_argument("--bg", default="data/background_images_train/Dalle_1.png", help="Path to background image")
    parser.add_argument("--font", default="ext/Microsoft Himalaya.ttf", help="Path to font file")
    parser.add_argument("--image_size", nargs=2, type=int, default=[768, 768], help="Image size (width height)")
    parser.add_argument("--box_pos", nargs=2, type=int, default=[200, 200], help="Bounding box position (x y)")
    parser.add_argument("--box_size", nargs=2, type=int, default=[300, 150], help="Initial bounding box size (width height)")
    parser.add_argument("--font_size", type=int, default=45, help="Font size")
    parser.add_argument("--text_source", choices=list(text_sources.keys()), default="corpus",
                        help=f"Text source type {text_sources.keys()}")
    parser.add_argument("--corpus_path", default="data/corpora/Tibetan Number Words/",
                        help="Path to corpus (if text_source is 'corpus')")
    parser.add_argument("--output", default="./test.png", help="Output image path")
    parser.add_argument("--augmentation", choices=list(augmentation_strategies.keys()), default='noise',
                        help="Type of augmentation to apply")

    return parser.parse_args()

def main():
    args = parse_args()

    # Generate text
    text_generator = text_sources[args.text_source](args.corpus_path)
    text = text_generator.generate_text()


    # The builder actually builds our synthetic image from different components
    builder = ImageBuilder(tuple(args.image_size))

    # retrofit the bounding box height to the actual height
    # when the text is wrapped by the initial bounding box' width.
    fitted_box_size = BoundingBoxCalculator.fit(text, tuple(args.box_size), font_size=args.font_size, font_path=args.font)

    augmentation = augmentation_strategies[args.augmentation]

    # construct our image
    image = (
        builder.set_background(args.bg)
        .set_font(args.font, font_size=args.font_size)
        .add_text(text, tuple(args.box_pos), fitted_box_size)
        .apply_augmentation(augmentation)
        .add_bounding_box(tuple(args.box_pos), fitted_box_size)
    )


    # Debugging
    image.show()
    image.save(args.output)

if __name__ == "__main__":
    main()