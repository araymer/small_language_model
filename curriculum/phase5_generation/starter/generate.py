import torch
import torch.nn.functional as F


def generate(model, tokenizer, prompt, max_new_tokens=200,
             temperature=1.0, top_k=None, device='cpu'):
    """
    Generate text from a prompt.

    1. Encode the prompt to token IDs
    2. Feed through the model
    3. Sample from the output distribution at the last position
    4. Append the new token and repeat
    5. Decode back to text

    Args:
        model: trained GPT model
        tokenizer: for encoding prompt and decoding output
        prompt: starting text string
        max_new_tokens: how many tokens to generate
        temperature: >1.0 = more random, <1.0 = more deterministic
        top_k: if set, only sample from the top k most likely tokens
    """
    raise NotImplementedError
