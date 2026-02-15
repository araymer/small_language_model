"""
Train a micrograd MLP on a simple toy problem.

Suggested: a tiny binary classification dataset, e.g.:
  inputs:  [( 2.0, 3.0), (-1.0, -2.0), ( 1.0, 1.0), (-2.0,  1.0)]
  targets: [         1.0,          -1.0,          1.0,          -1.0]

Training loop:
  1. Forward pass: run each input through the model
  2. Compute loss (e.g., hinge loss or MSE)
  3. Zero gradients
  4. Backward pass
  5. Update parameters (gradient descent: p.data -= learning_rate * p.grad)
  6. Repeat for N iterations, print loss each step
"""
from micrograd.nn import MLP


def train():
    raise NotImplementedError


if __name__ == '__main__':
    train()
