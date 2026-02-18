class BPETokenizer:
    """
    Byte Pair Encoding tokenizer.

    Starts with character-level tokens and iteratively merges
    the most frequent adjacent pairs into new tokens.
    """

    def __init__(self):
        self.merges = {}        # (id1, id2) -> merged_id
        self.vocab = {}         # id -> string
        self.inverse_vocab = {} # string -> id

    def train(self, text, vocab_size):
        """
        Learn BPE merges from training text.

        1. Start with character-level vocabulary (sorted unique chars)
        2. Convert text to list of character-level token IDs
        3. Count all adjacent token pairs
        4. Merge the most frequent pair into a new token
        5. Repeat until desired vocab_size is reached
        """
        raise NotImplementedError

    def encode(self, text):
        """
        Apply learned merges to encode text.

        1. Split text into characters, convert to base IDs
        2. Apply each learned merge in order
        """
        raise NotImplementedError

    def decode(self, token_ids):
        """Reverse the encoding: map IDs to strings, concatenate."""
        raise NotImplementedError

    @property
    def vocab_size(self):
        """Number of unique tokens in the vocabulary."""
        raise NotImplementedError

    def _count_pairs(self, tokens):
        """Count all adjacent pairs in a token list. Returns dict of pair -> count."""
        raise NotImplementedError

    def _apply_merge(self, tokens, pair, new_id):
        """Replace all occurrences of pair in tokens with new_id."""
        raise NotImplementedError
