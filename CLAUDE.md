# SML Project Context

## What This Is
A learning project to build a GPT-style transformer language model from scratch. The goal is deep understanding of how LLMs work, not production quality. Aaron is implementing the educational/interesting parts himself; Claude handles boilerplate, tooling, config, and acts as mentor/co-partner.

## Roles
- **Aaron implements**: Tokenizer, attention mechanism, transformer blocks, positional encoding, model architecture, training loop, generation logic, anything with real learning value
- **Claude handles**: Project setup, requirements, git workflow, debugging assistance, code review, explaining concepts, writing tests/utilities when asked
- **Key rule**: Don't write the interesting parts for Aaron. Guide, explain, review — but let him write the code that matters for learning.

## Hardware
- **MacBook**: 24GB RAM, CPU only (Apple Silicon). For development, validation, debugging. No training here.
- **PC**: 16GB RTX GPU (CUDA). For actual training. Not yet set up for this project.

## Tech Stack
- Python 3.13.5
- PyTorch 2.10 (tensors, autograd, CUDA abstraction)
- NumPy 2.4 (micrograd phase, utilities)
- Matplotlib 3.10 (loss curves, visualization)

## Architecture (Planned)
- ~6 attention heads
- ~6 transformer blocks
- 384 embedding dimensions
- ~256 token context window
- ~10-25M parameters (depending on vocab size)

## Project Structure
```
SML/
├── CLAUDE.md
├── README.md
├── requirements.txt
└── (to be built out as we go)
```

## Conventions
- Keep it simple. No over-engineering.
- Flat structure until complexity demands otherwise.
- Explicit over clever. This is for learning, not golf.

---

## Syllabus

### Phase 1: Micrograd — Autograd & Backprop Fundamentals
- [x] Implement `Value` class with basic ops (+, *, etc.)
- [x] Implement backward pass (reverse-mode autodiff)
- [x] Verify gradients against PyTorch
- [x] Build a simple neuron / MLP using Value
- [x] Train on a toy problem (e.g., simple classification)
- [ ] ~~Visualize the computation graph~~ (skipped)
- [x] **Checkpoint**: Understand chain rule, computational graphs, gradient flow

### Phase 2: Tokenizer
- [ ] ~~Character-level tokenizer~~ (skipped — went straight to BPE)
- [x] Load and preprocess a toy dataset (tiny Shakespeare)
- [x] Byte Pair Encoding (BPE) tokenizer from scratch
- [x] Vocab building and special tokens
- [x] **Checkpoint**: Understand how text becomes numbers for a model

### Phase 3: Transformer Architecture
- [ ] Token embeddings and positional encodings
- [ ] Scaled dot-product attention (single head)
- [ ] Multi-head attention
- [ ] Feed-forward network
- [ ] Layer normalization
- [ ] Residual connections
- [ ] Full transformer block
- [ ] Assemble complete GPT model
- [ ] **Checkpoint**: Forward pass works, model can be loaded into memory on MacBook

### Phase 4: Training Pipeline
- [ ] Dataset / DataLoader for text sequences
- [ ] Training loop with cross-entropy loss
- [ ] Learning rate scheduling
- [ ] Loss tracking and plotting
- [ ] Checkpointing (save/load model state)
- [ ] Validation split and basic eval
- [ ] **Checkpoint**: Model trains on toy data, loss goes down

### Phase 5: Text Generation
- [ ] Greedy decoding
- [ ] Temperature sampling
- [ ] Top-k sampling
- [ ] Generate coherent(ish) text from trained model
- [ ] **Checkpoint**: Model produces recognizable English from tiny Shakespeare

### Phase 6: Scale Up & GPU Training
- [ ] Set up PC environment with CUDA
- [ ] Verify model trains on GPU
- [ ] Train on larger dataset / more epochs
- [ ] Hyperparameter tuning
- [ ] Evaluate generation quality
- [ ] **Checkpoint**: Properly trained model generating decent text

---

## Current Status
**Phase**: Phase 3 — Transformer Architecture
**Last completed**: Phase 2 complete (BPE tokenizer, tiny Shakespeare dataset downloaded)
