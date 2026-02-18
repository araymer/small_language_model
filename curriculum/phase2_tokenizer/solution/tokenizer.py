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
        chars = sorted(set(text))
        self.vocab = {i: c for i, c in enumerate(chars)}
        self.inverse_vocab = {c: i for i, c in enumerate(chars)}

        tokens = [self.inverse_vocab[c] for c in text]

        while len(self.vocab) < vocab_size:
            pairs = self._count_pairs(tokens)
            if not pairs:
                break

            best_pair = max(pairs, key=pairs.get)
            new_id = len(self.vocab)

            self.merges[best_pair] = new_id
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            self.inverse_vocab[self.vocab[new_id]] = new_id

            tokens = self._apply_merge(tokens, best_pair, new_id)

    def encode(self, text):
        tokens = [self.inverse_vocab[c] for c in text]

        for pair, new_id in self.merges.items():
            tokens = self._apply_merge(tokens, pair, new_id)

        return tokens

    def decode(self, token_ids):
        return ''.join(self.vocab[id] for id in token_ids)

    @property
    def vocab_size(self):
        return len(self.vocab)

    def _count_pairs(self, tokens):
        counts = {}
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def _apply_merge(self, tokens, pair, new_id):
        merged = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                merged.append(new_id)
                i += 2
            else:
                merged.append(tokens[i])
                i += 1
        return merged
