# Phase 1: Autograd — Understanding Backpropagation

## What You'll Learn

- How automatic differentiation (autograd) works under the hood
- The chain rule applied as a computational graph traversal
- How neural networks are composed from simple differentiable operations
- The complete training loop: forward pass, loss, backward pass, weight update

## Concepts

### Computational Graphs
Every math operation creates a node in a directed acyclic graph. The graph records
what operations happened and what values were involved. This is the "tape" that
autograd uses to compute gradients.

### The Chain Rule
If `c = a * b` and `d = c + e`, the gradient of `d` with respect to `a` is:
`dd/da = dd/dc * dc/da = 1 * b = b`

Each node only needs to know its own local derivative. The graph handles chaining
them together automatically.

### Gradient Accumulation
Gradients use `+=` not `=` because a value used in multiple operations receives
gradient contributions from each path. The total gradient is the sum.

### The Training Loop
1. **Forward pass** — compute predictions and loss (builds the graph)
2. **Zero gradients** — clear stale gradients from previous iteration
3. **Backward pass** — walk the graph in reverse, fill in gradients via chain rule
4. **Update weights** — nudge each weight opposite to its gradient (gradient descent)

## Files to Implement

### `engine.py` — The Value Class (start here)
The core autograd engine. A `Value` wraps a scalar and tracks the computation graph.

**Implementation order:**
1. `__add__` and `__mul__` — basic ops with `_backward` functions
2. `__pow__` — power rule: derivative of `x^n` is `n * x^(n-1)`
3. `__neg__`, `__sub__`, `__truediv__` — built from existing ops
4. `__radd__`, `__rmul__` — handle `number + Value` ordering
5. `tanh()`, `relu()`, `exp()` — activation functions with their derivatives
6. `backward()` — topological sort, then walk in reverse calling `_backward`

**Key pattern for every operation:**
```python
def __add__(self, other):
    out = Value(self.data + other.data, (self, other), '+')
    def _backward():
        self.grad += out.grad    # local derivative * incoming gradient
        other.grad += out.grad
    out._backward = _backward
    return out
```

### `nn.py` — Neural Network Building Blocks
- `Neuron`: weighted sum of inputs + bias, then activation
- `Layer`: a collection of neurons receiving the same inputs
- `MLP`: a stack of layers

### `verify_gradients.py` — Sanity Check
Build the same expression in both your Value class and PyTorch tensors.
Compare gradients — they should match within floating point tolerance.

### `train_toy.py` — The Payoff
Train an MLP on 4 data points. Watch the loss go down.
This is the same loop every neural network uses, just at tiny scale.

## Running

```bash
# Verify gradients match PyTorch
python verify_gradients.py

# Train on toy data
python train_toy.py
```

## Key Takeaways

- PyTorch's `loss.backward()` does exactly what your `backward()` does — just on tensors instead of scalars
- PyTorch's `optimizer.step()` does exactly what your `p.data -= lr * p.grad` loop does
- The entire training process is: predict → measure error → compute gradients → adjust weights → repeat
