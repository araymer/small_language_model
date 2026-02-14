# Small Language Model (SML)

A GPT-style transformer language model implemented from scratch for learning purposes.

## Goals

- Implement a multi-head attention transformer from the ground up
- Understand tokenization, attention, positional encoding, and text generation
- Train on toy datasets locally, scale to GPU for real training

## Architecture (Planned)

- ~6 attention heads
- ~6 transformer blocks
- 384 embedding dimensions
- ~256 token context window

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Phases

1. **Micrograd** - Tiny autograd engine to understand backpropagation
2. **Tokenizer** - Character-level, then BPE
3. **Transformer** - Full model architecture in PyTorch
4. **Training** - Toy data first, then scale up
5. **Evaluation** - Tuning, generation quality, cross-validation
