import random
from engine import Value


class Neuron:
    """
    A single neuron: computes weighted sum of inputs + bias, then activation.

    output = activation(w1*x1 + w2*x2 + ... + wn*xn + bias)
    """

    def __init__(self, n_inputs, activation='tanh'):
        """Initialize with random weights and zero bias."""
        raise NotImplementedError

    def __call__(self, x):
        """Forward pass: dot product + bias, then activation."""
        raise NotImplementedError

    def parameters(self):
        """Return list of all tunable Values (weights + bias)."""
        raise NotImplementedError


class Layer:
    """
    A layer of neurons. Each neuron receives the same inputs.
    """

    def __init__(self, n_inputs, n_outputs, **kwargs):
        """Create n_outputs neurons, each with n_inputs inputs."""
        raise NotImplementedError

    def __call__(self, x):
        """Forward pass: run each neuron on the input."""
        raise NotImplementedError

    def parameters(self):
        """Return all parameters from all neurons."""
        raise NotImplementedError


class MLP:
    """
    Multi-layer perceptron: a stack of Layers.

    Example: MLP(3, [4, 4, 1]) creates:
      - Layer 1: 3 inputs -> 4 outputs
      - Layer 2: 4 inputs -> 4 outputs
      - Layer 3: 4 inputs -> 1 output
    """

    def __init__(self, n_inputs, layer_sizes):
        """Build the stack of layers."""
        raise NotImplementedError

    def __call__(self, x):
        """Forward pass: run input through each layer sequentially."""
        raise NotImplementedError

    def parameters(self):
        """Return all parameters from all layers."""
        raise NotImplementedError
