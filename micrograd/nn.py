import random
from micrograd.engine import Value


class Neuron:
    """
    A single neuron: computes weighted sum of inputs + bias, then activation.

    output = activation(w1*x1 + w2*x2 + ... + wn*xn + bias)
    """

    def __init__(self, n_inputs, activation='tanh'):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_inputs)]
        self.b = Value(0.0)
        self.activation = activation

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        if self.activation == 'tanh':
            return act.tanh()
        elif self.activation == 'relu':
            return act.relu()
        return act

    def parameters(self):
        return self.w + [self.b]


class Layer:
    """
    A layer of neurons. Each neuron receives the same inputs.
    """

    def __init__(self, n_inputs, n_outputs, **kwargs):
        self.neurons = [Neuron(n_inputs, **kwargs) for _ in range(n_outputs)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP:
    """
    Multi-layer perceptron: a stack of Layers.

    Example: MLP(3, [4, 4, 1]) creates:
      - Layer 1: 3 inputs -> 4 outputs
      - Layer 2: 4 inputs -> 4 outputs
      - Layer 3: 4 inputs -> 1 output
    """

    def __init__(self, n_inputs, layer_sizes):
        sizes = [n_inputs] + layer_sizes
        self.layers = [Layer(sizes[i], sizes[i + 1]) for i in range(len(layer_sizes))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
