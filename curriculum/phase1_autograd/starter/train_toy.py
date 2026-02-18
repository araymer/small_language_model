"""
Train a micrograd MLP on a simple toy problem.

Dataset: 4 input/target pairs for binary classification.
The network should memorize them — overfitting is the goal here.

Training loop:
  1. Forward pass: run each input through the model
  2. Compute loss (MSE: sum of squared errors)
  3. Zero gradients
  4. Backward pass
  5. Update parameters (gradient descent: p.data -= learning_rate * p.grad)
  6. Repeat for N iterations, print loss each step
"""
from engine import Value
from nn import MLP

# Arbitrary data — we just want to see the network learn
xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
ys = [1.0, -1.0, -1.0, 1.0]

train_set = list(zip(xs, ys))


def train(iterations=100, learning_rate=0.05):
    raise NotImplementedError


if __name__ == '__main__':
    train()
