import torch
from sml.model import GPT


def get_batch(data, context_length, batch_size, device='cpu'):
    """
    Sample a random batch of training examples.

    Pick random starting positions in the data, then slice out
    context_length chunks for input (x) and targets (y = x shifted by 1).

    Returns:
        x: (batch_size, context_length) tensor of token IDs
        y: (batch_size, context_length) tensor of target token IDs
    """
    raise NotImplementedError


def train(model, train_data, val_data, config):
    """
    Training loop.

    For each iteration:
      1. Sample a batch
      2. Forward pass (get logits and loss)
      3. Backward pass
      4. Update weights (optimizer.step())
      5. Periodically evaluate on validation data
      6. Periodically save checkpoints
      7. Log/plot training loss
    """
    raise NotImplementedError


@torch.no_grad()
def estimate_loss(model, data, config):
    """
    Estimate average loss over multiple batches.

    Used for validation — gives a more stable loss estimate
    than looking at a single batch.
    """
    raise NotImplementedError
