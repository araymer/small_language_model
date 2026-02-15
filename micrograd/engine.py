import math


class Value:
    """
    Wraps a scalar value and tracks the computation graph for autograd.

    Every Value knows:
    - its data (the actual number)
    - its gradient (how much the final output changes if this value changes)
    - its children (what Values were used to create it)
    - its _backward function (how to propagate gradients to its children)
    """

    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None


    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"


    def __add__(self, other):
        """Add two Values (or a Value and a number). Track in the graph."""
        if type(other) is not Value:
            other = Value(other)

        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward

        return out


    def __mul__(self, other):
        """Multiply two Values (or a Value and a number). Track in the graph."""
        if type(other) is not Value:
            other = Value(other)

        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data

        out._backward = _backward

        return out


    def __pow__(self, other):
        """Raise a Value to a numeric power. Track in the graph."""

        out =  Value(self.data ** other, (self,), '**')

        def _backward():
            self.grad += other * (self.data ** (other-1)) * out.grad

        out._backward = _backward

        return out


    def __neg__(self):
        """Negate a Value. Hint: this can be built from ops you already have."""
        return self * Value(-1)


    def __sub__(self, other):
        """Subtract. Hint: this can be built from ops you already have."""
        return self + (-other)


    def __truediv__(self, other):
        """Divide. Hint: this can be built from ops you already have."""
        return self * other**(-1)


    def __radd__(self, other):
        """Handles: number + Value (e.g., 2 + Value(3))."""
        return self + other


    def __rmul__(self, other):
        """Handles: number * Value (e.g., 2 * Value(3))."""
        return self * other


    def tanh(self):
        """Hyperbolic tangent activation. Track in the graph."""

        out = Value(math.tanh(self.data), (self,), 'tanh')

        def _backward():
            self.grad += (1 - out.data**2) * out.grad

        out._backward = _backward

        return out


    def relu(self):
        """ReLU activation. Track in the graph."""
        out = Value(max(self.data,0), (self,), 'relu')

        def _backward():
            self.grad += (1 if out.data > 0 else 0) * out.grad

        out._backward = _backward

        return out


    def exp(self):
        """Exponential. Track in the graph."""
        out = Value(math.exp(self.data), (self,), 'exp')

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward

        return out


    def backward(self):
        topo = []
        visited = set()

        def build_topo(node):
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    build_topo(child)
                topo.append(node)

        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
