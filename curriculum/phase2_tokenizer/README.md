# Phase 2: Tokenizer — From Text to Numbers

## What You'll Learn

- How text is converted into sequences of integers for a model
- Byte Pair Encoding (BPE) — the tokenization algorithm used by GPT
- Why subword tokenization is better than character-level or word-level
- The compression vs. vocabulary size tradeoff

## Concepts

### Why Tokenize?
Neural networks operate on numbers, not text. A tokenizer is the bridge: it maps
text to integer sequences (encoding) and back (decoding).

### Character-Level vs. Word-Level vs. Subword
- **Character-level**: Tiny vocabulary (~100 tokens), but sequences are very long.
  "hello" = 5 tokens. Simple but inefficient.
- **Word-level**: Short sequences, but massive vocabulary. Can't handle new words.
  Every typo or rare word is unknown.
- **Subword (BPE)**: Best of both worlds. Common words get their own token,
  rare words are split into known pieces. "unhappiness" → ["un", "happiness"]

### BPE Algorithm
1. Start with character-level vocabulary
2. Count all adjacent token pairs in the corpus
3. Merge the most frequent pair into a new token
4. Repeat until target vocabulary size is reached

The merge order matters — it must be replayed in the same order during encoding.

## Files to Implement

### `tokenizer.py` — BPE Tokenizer

**`train(text, vocab_size)`**: Learn merges from a corpus
- Build character vocabulary
- Convert text to character-level token IDs
- Loop: count pairs → find most frequent → merge → repeat

**`encode(text)`**: Convert text to token IDs
- Split into characters, map to base IDs
- Apply learned merges in order

**`decode(token_ids)`**: Convert token IDs back to text
- Map each ID to its string, concatenate

**Helper functions:**
- `_count_pairs(tokens)`: Count adjacent pairs in a token list
- `_apply_merge(tokens, pair, new_id)`: Replace all occurrences of a pair

## Testing

```python
tok = BPETokenizer()
tok.train("the cat sat on the mat", vocab_size=30)
encoded = tok.encode("the cat")
assert tok.decode(encoded) == "the cat"  # round-trip must work
```

## Key Takeaways

- Tokenization is a fixed preprocessing step — train once, use forever
- BPE naturally handles any text, including words it hasn't seen before
- Vocabulary size is a hyperparameter: bigger = shorter sequences but more embeddings
- Real tokenizers (GPT-4: ~100K tokens) are trained on massive corpora
