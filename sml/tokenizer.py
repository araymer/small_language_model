class Tokenizer:
    """
    Converts text to sequences of integers and back.

    Phase 1: Character-level (each unique character = one token)
    Phase 2: Byte Pair Encoding (BPE) for subword tokenization
    """

    def __init__(self):
        """Build vocabulary from training text."""
        raise NotImplementedError

    def encode(self, text):
        """Convert a string to a list of integer token IDs."""
        raise NotImplementedError

    def decode(self, token_ids):
        """Convert a list of integer token IDs back to a string."""
        raise NotImplementedError

    @property
    def vocab_size(self):
        """Number of unique tokens in the vocabulary."""
        raise NotImplementedError


class BPETokenizer(Tokenizer):
    """
    Byte Pair Encoding tokenizer.

    Starts with character-level tokens and iteratively merges
    the most frequent adjacent pairs into new tokens.
    """

    def train(self, text, vocab_size):
        """
        Learn BPE merges from training text.

        1. Start with character-level vocabulary
        2. Count all adjacent token pairs
        3. Merge the most frequent pair into a new token
        4. Repeat until desired vocab_size is reached
        """
        raise NotImplementedError

    def encode(self, text):
        """Apply learned merges to encode text."""
        raise NotImplementedError

    def decode(self, token_ids):
        """Reverse the encoding back to text."""
        raise NotImplementedError
