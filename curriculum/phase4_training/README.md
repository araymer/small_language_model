# Phase 4: Training Pipeline — Making the Model Learn

## What You'll Learn

- How to prepare text data for transformer training
- Batching and the input/target relationship (shifted by one)
- The training loop with PyTorch optimizers
- Validation, checkpointing, and loss tracking

## Concepts

### Data Preparation
The training data is one long sequence of token IDs. We sample random chunks
of `context_length` tokens as input, with the target being the same chunk
shifted right by one position.

```
Text: "The quick brown fox jumps"
Input:  [The, quick, brown, fox]
Target: [quick, brown, fox, jumps]
```

### Batching
Instead of one example per weight update, we process a batch of examples
simultaneously. This gives smoother gradients and keeps the GPU busy.

### The Training Loop
Same concept as Phase 1, but now with PyTorch handling the autograd:
1. Sample a batch of (input, target) pairs
2. Forward pass: model produces logits and cross-entropy loss
3. `loss.backward()` — PyTorch computes all gradients
4. `optimizer.step()` — updates weights (Adam optimizer)
5. `optimizer.zero_grad()` — clears gradients for next iteration

### Validation
Periodically estimate loss on held-out data to detect overfitting.
Use `@torch.no_grad()` to skip gradient computation during evaluation.

### Checkpointing
Save model weights periodically so you can resume training or use
the best checkpoint for generation.

## Files to Implement

### `train.py`
- `get_batch()`: Sample random chunks from the data
- `train()`: The main training loop
- `estimate_loss()`: Average loss over multiple batches for stable evaluation

## Key Takeaways

- The model gets N learning signals per example (one per position in the context)
- Adam optimizer is standard — it adapts learning rate per-parameter
- Validation loss tells you when to stop; training loss tells you if learning is working
