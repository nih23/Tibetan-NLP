from PIL import ImageFont, ImageDraw, Image

class BoundingBoxCalculator:
    @staticmethod
    def calculate(text, font, box_position, box_size):
        """
        Berechnet die Bounding-Box für den gegebenen Text.

        :param text: Der Text, für den die Bounding-Box berechnet wird.
        :param font: Pfad zur Schriftartdatei.
        :param box_position: (x, y) Position des Textfelds.
        :param box_size: (width, height) des Textfelds.
        :return: Liste von Bounding-Boxen [(x, y, width, height), ...].
        """
        # Initialisiere die Schriftart
        try:
            font = ImageFont.truetype(font, size=24)
        except IOError:
            raise ValueError(f"Font at {font} could not be loaded.")

        # Dummy-Image für Textmessung
        dummy_image = Image.new('RGB', (1, 1))
        draw = ImageDraw.Draw(dummy_image)

        # Box-Parameter
        box_x, box_y = box_position
        box_w, box_h = box_size

        # Resultierende Bounding-Boxen
        bounding_boxes = []
        y_offset = 0

        for line in text.split('\n'):
            while line:
                # Textzeilen umbrechen
                for i in range(1, len(line) + 1):
                    if draw.textlength(line[:i], font=font) > box_w:
                        break
                else:
                    i = len(line)

                # Berechne die Bounding-Box der aktuellen Zeile
                wrapped_line = line[:i]
                left, top, right, bottom = font.getbbox(wrapped_line)

                # Füge die Box hinzu, falls sie in den Rahmen passt
                if y_offset + (bottom - top) <= box_h:
                    bounding_boxes.append((
                        box_x,  # Start x
                        box_y + y_offset,  # Start y
                        right - left,  # Breite
                        bottom - top  # Höhe
                    ))
                    y_offset += (bottom - top)
                else:
                    break  # Kein Platz mehr für weitere Zeilen

                # Schneide die verarbeitete Zeile ab
                line = line[i:]

        return bounding_boxes