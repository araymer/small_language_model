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
from micrograd import Value
from micrograd.nn import MLP

# Just some trash data for toying with
xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
ys = [1.0, -1.0, -1.0, 1.0]

train_set = list(zip(xs, ys))

def train(iterations=100, learning_rate=0.01):
    nn = MLP(3, [4, 4, 1])

    for i in range(iterations):
        loss = 0
        for x, y in train_set:
            x, y = [Value(i) for i in x], Value(y)
            pred = nn(x)
            loss += (pred-y)**2
        for p in nn.parameters():
            p.grad = 0
        loss.backward()
        for p in nn.parameters():
            p.data -= p.grad * learning_rate
        print(f"iteration {i}: loss = {loss.data:.6f}")









if __name__ == '__main__':
    train(iterations=15000, learning_rate=.1)
