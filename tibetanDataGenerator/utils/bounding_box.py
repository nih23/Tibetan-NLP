from PIL import ImageFont, ImageDraw, Image

class BoundingBoxCalculator:
    @staticmethod
    def fit(text, box_size, font_size = 24, font_path ='res/Microsoft Himalaya.ttf'):
        """
        Calculate the true bounding box size for the specified text when it is wrapped and terminated to fit a given box size.

        :param text: Text to be measured.
        :param box_size: Tuple (width, height) specifying the size of the box to fit the text.
        :param font_size: Size of the font.
        :param font_path: Path to the font file.
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

        return actual_text_width, y_offset+10