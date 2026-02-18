# Phase 6: GPU Scaling — Training for Real

## What You'll Learn

- Setting up CUDA for GPU-accelerated training
- Moving model and data to GPU with `.to("cuda")`
- Training on larger datasets with more epochs
- Hyperparameter tuning and evaluating generation quality

## Concepts

### CPU vs GPU
Everything we've built runs on CPU by default. PyTorch makes the switch
to GPU nearly transparent — move the model and data to CUDA, and all
operations run on the GPU automatically.

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = GPT(...).to(device)
x, y = get_batch(data, ..., device=device)
```

### What Changes on GPU
- Matrix multiplications run 10-100x faster
- Training that took hours on CPU takes minutes
- Batch sizes can be larger (more GPU memory available)
- Everything else stays the same — same model, same code

### Hyperparameter Tuning
With faster training, you can experiment with:
- Learning rate and schedule
- Batch size
- Number of layers and heads
- Embedding dimension
- Context length
- Dropout rate

### Evaluation
- Loss curves (training vs validation)
- Qualitative evaluation (does the generated text make sense?)
- Perplexity as a quantitative metric

## Setup

```bash
# On your PC with NVIDIA GPU
pip install torch --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.cuda.is_available())"  # should print True
```

## Key Takeaways

- GPU training is the same code, just faster
- The model architecture doesn't change — only the device
- Real training is mostly about patience, data quality, and hyperparameter tuning
