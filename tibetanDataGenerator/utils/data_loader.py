import abc
import os
import random
from abc import abstractclassmethod


class TextFactory:
    @staticmethod
    def create_text_source(source_type, *args, **kwargs):
        if source_type == "synthetic":
            return SyntheticTextGenerator(*args, **kwargs)
        elif source_type == "corpus":
            return CorpusTextGenerator(*args, **kwargs)
        else:
            raise ValueError(f"Unknown source type: {source_type}")

class TextGenerator:
    @staticmethod
    def generate_random_text(length):
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

    @abc.abstractmethod
    def generate_text(self):
        pass

class SyntheticTextGenerator(TextGenerator):
    def generate_text(self):
        """
        Generate a lorem ipsum like Tibetan text string of a specified length.

        This function creates words of random lengths and separates them with a space,
        similar to the structure of lorem ipsum text.
        """
        tibetan_range = (0x0F40, 0x0FBC)  # Restricting range to more common characters
        no_words = random.randint(1,50)
        word_lengths = [random.randint(2, 10) for _ in range(no_words)]

        words = []
        for word_length in word_lengths:
            word = ''.join(chr(random.randint(*tibetan_range)) for _ in range(word_length))
            words.append(word)

        return ' '.join(words)

class CorpusTextGenerator(TextGenerator):
    def __init__(self, corpus_dir):
        self.corpus_dir = corpus_dir

    def generate_text(self):
        # Wähle zufälligen Text aus dem Corpus
        text_to_embed, filename = self._read_random_tibetan_file()
        return text_to_embed

    def _read_random_tibetan_file(self):
        """
        Read a random text file containing Tibetan text from a specified directory.

        :param directory: The directory containing Tibetan text files.
        :return: Content of a randomly selected text file.
        """
        # List all files in the specified directory
        files = [f for f in os.listdir(self.corpus_dir) if os.path.isfile(os.path.join(self.corpus_dir, f))]
        if not files:
            return "No files found in the specified directory."

        # Randomly select a file
        random_file = random.choice(files)
        file_path = os.path.join(self.corpus_dir, random_file)

        # Read the content of the file
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
        except Exception as e:
            return f"Error reading file {random_file}: {e}"

        return content, random_file