import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """
    Single head of scaled dot-product self-attention.

    1. Project input to queries, keys, and values
    2. Compute attention scores: (Q @ K^T) / sqrt(d_k)
    3. Apply causal mask (prevent looking at future tokens)
    4. Softmax to get attention weights
    5. Weighted sum of values
    """

    def __init__(self, embed_dim, head_dim):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


class MultiHeadAttention(nn.Module):
    """
    Multiple attention heads in parallel, then concatenate and project.
    """

    def __init__(self, embed_dim, n_heads):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.

    Two linear layers with a non-linearity in between.
    Typically expands the dimension by 4x, then projects back down.
    """

    def __init__(self, embed_dim):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


class TransformerBlock(nn.Module):
    """
    One transformer block:
      x -> LayerNorm -> MultiHeadAttention -> residual add
        -> LayerNorm -> FeedForward -> residual add
    """

    def __init__(self, embed_dim, n_heads):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


class GPT(nn.Module):
    """
    The full GPT model:
      1. Token embedding + positional encoding
      2. Stack of TransformerBlocks
      3. Final LayerNorm
      4. Linear projection to vocab_size (logits)
    """

    def __init__(self, vocab_size, embed_dim=384, n_heads=6,
                 n_layers=6, context_length=256):
        super().__init__()
        raise NotImplementedError

    def forward(self, token_ids, targets=None):
        """
        Forward pass.

        Args:
            token_ids: (batch, sequence_length) integer tensor
            targets: (batch, sequence_length) integer tensor, optional

        Returns:
            logits: (batch, sequence_length, vocab_size)
            loss: cross-entropy loss if targets provided, else None
        """
        raise NotImplementedError

    def generate(self, token_ids, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate new tokens autoregressively.

        Repeatedly: forward pass → sample from last position → append → repeat
        """
        raise NotImplementedError
