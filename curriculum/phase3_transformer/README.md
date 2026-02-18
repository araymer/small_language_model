# Phase 3: Transformer Architecture — The Heart of GPT

## What You'll Learn

- How self-attention lets tokens communicate with each other
- Why multi-head attention works better than single-head
- The role of feed-forward networks, layer normalization, and residual connections
- How all the pieces assemble into a complete GPT model

## Concepts

### Self-Attention
The mechanism that lets each token look at other tokens and decide which ones
are relevant. Each token's embedding is projected into three vectors:
- **Query (Q)**: "What am I looking for?"
- **Key (K)**: "What do I contain?"
- **Value (V)**: "What information do I provide?"

Attention scores: `softmax(Q @ K^T / sqrt(d_k)) @ V`

### Causal Masking
During training, the model predicts all positions simultaneously. The causal mask
prevents each position from attending to future tokens — it zeros out the upper
triangle of the attention matrix. This is what makes it autoregressive.

### Multi-Head Attention
Instead of one big attention operation, run several smaller ones in parallel.
Each head can learn to attend to different things (syntax, semantics, position, etc.).
Concatenate the results and project back to the original dimension.

### Residual Connections
`output = x + sublayer(x)` — the input is added back to the output of each sublayer.
This helps gradients flow through deep networks and lets layers learn incremental
modifications rather than complete transformations.

### Layer Normalization
Normalizes activations to have zero mean and unit variance. Stabilizes training
and helps the model converge faster.

### Feed-Forward Network
A simple two-layer MLP applied independently to each position. Expands the
dimension by 4x, applies a non-linearity, then projects back down. This is
where most of the model's parameters live.

### The Full Transformer Block
```
x → LayerNorm → MultiHeadAttention → + (residual) → LayerNorm → FeedForward → + (residual)
```

### GPT Assembly
```
Token IDs → Token Embedding + Positional Encoding
          → Stack of Transformer Blocks
          → Final LayerNorm
          → Linear projection to vocab_size (logits)
```

## Files to Implement

### `model.py` — implement in this order:

1. **SelfAttention**: Single attention head (Q, K, V projections, masking, softmax)
2. **MultiHeadAttention**: Multiple heads in parallel, concatenate, project
3. **FeedForward**: Two linear layers with GELU activation
4. **TransformerBlock**: LayerNorm + attention + residual + LayerNorm + FFN + residual
5. **GPT**: Embeddings, stack of blocks, final projection

## Key Takeaways

- Attention is just a weighted average — the weights are learned from the data
- The causal mask is what makes the model autoregressive, not the computation order
- Residual connections and layer norm are what make deep transformers trainable
- The entire model is differentiable — PyTorch handles backprop through all of it
