"""
Verify micrograd gradients against PyTorch.

Build the same expression in both systems, run backward,
and compare the gradients to make sure they match.
"""
import torch
from micrograd.engine import Value


def verify():
    # -- micrograd --
    # TODO: Build an expression using Value, e.g.:
    # a = Value(2.0)
    # b = Value(-3.0)
    # c = a * b + a ** 2
    # c.backward()

    # -- pytorch --
    # TODO: Build the same expression with torch tensors (requires_grad=True)
    # Compare .grad values — they should match

    raise NotImplementedError


if __name__ == '__main__':
    verify()
    print("All gradients match!")
