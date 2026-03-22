from textblob import TextBlob


class WordBuilder:
    def __init__(self):
        self.current_word = ""
        self.last_char = ""
        self.repeat_count = 0

    def update(self, prediction):
        """
        Update word based on prediction
        """
        if prediction is None:
            return self.current_word

        # Reduce noise (same character repeated)
        if prediction == self.last_char:
            self.repeat_count += 1
        else:
            self.repeat_count = 0

        # Add character only if stable (detected multiple times)
        if self.repeat_count == 5:
            self.current_word += prediction
            self.repeat_count = 0

        self.last_char = prediction
        return self.current_word

    def get_corrected_word(self):
        """
        Apply autocorrect
        """
        if not self.current_word:
            return ""

        blob = TextBlob(self.current_word)
        return str(blob.correct())

    def reset(self):
        """
        Reset the current word
        """
        self.current_word = ""
        self.last_char = ""
        self.repeat_count = 0