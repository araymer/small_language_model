"""
Verify micrograd gradients against PyTorch.

Build the same expression in both systems, run backward,
and compare the gradients to make sure they match.
"""
import torch
from engine import Value


def verify():
    # -- micrograd --
    # TODO: Build an expression using Value, e.g.:
    # a = Value(2.0)
    # b = Value(-3.0)
    # Use add, mul, pow, and tanh to create a multi-step expression
    # Call backward() on the final result

    # -- pytorch --
    # TODO: Build the same expression with torch tensors (requires_grad=True)
    # Call backward() on the final result
    # Compare .grad values — they should match within 1e-6

    raise NotImplementedError


if __name__ == '__main__':
    verify()
    print("\nAll gradients match!")
