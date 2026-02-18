# Phase 5: Text Generation — Making the Model Talk

## What You'll Learn

- How autoregressive generation works (predict one, append, repeat)
- Temperature and its effect on output randomness
- Top-k sampling for controlling generation quality
- The difference between greedy decoding and sampling

## Concepts

### Autoregressive Generation
The model generates one token at a time:
1. Encode the prompt to token IDs
2. Forward pass — get probability distribution over next token
3. Sample from that distribution
4. Append the new token to the sequence
5. Repeat from step 2

### Temperature
Scales the logits before softmax. Controls randomness:
- **temperature = 1.0**: Normal sampling from the learned distribution
- **temperature < 1.0**: Sharper distribution, more deterministic (greedy at 0)
- **temperature > 1.0**: Flatter distribution, more random and creative

### Top-k Sampling
Only sample from the k most likely tokens. Prevents the model from
occasionally picking very unlikely tokens that derail the output.

### Greedy Decoding
Always pick the most likely token (equivalent to temperature → 0).
Produces repetitive but "safe" output.

## Files to Implement

### `generate.py`
- Encode prompt, run generation loop, decode back to text
- Implement temperature scaling and top-k filtering

## Key Takeaways

- Generation is sequential — each token depends on the actual previous token
- This is where KV cache matters for speed (not needed for correctness)
- The same model can produce wildly different output with different temperature/top-k
