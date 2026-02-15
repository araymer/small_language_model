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
        raise NotImplementedError

    def __mul__(self, other):
        """Multiply two Values (or a Value and a number). Track in the graph."""
        raise NotImplementedError

    def __pow__(self, other):
        """Raise a Value to a numeric power. Track in the graph."""
        raise NotImplementedError

    def __neg__(self):
        """Negate a Value. Hint: this can be built from ops you already have."""
        raise NotImplementedError

    def __sub__(self, other):
        """Subtract. Hint: this can be built from ops you already have."""
        raise NotImplementedError

    def __truediv__(self, other):
        """Divide. Hint: this can be built from ops you already have."""
        raise NotImplementedError

    def __radd__(self, other):
        """Handles: number + Value (e.g., 2 + Value(3))."""
        raise NotImplementedError

    def __rmul__(self, other):
        """Handles: number * Value (e.g., 2 * Value(3))."""
        raise NotImplementedError

    def tanh(self):
        """Hyperbolic tangent activation. Track in the graph."""
        raise NotImplementedError

    def relu(self):
        """ReLU activation. Track in the graph."""
        raise NotImplementedError

    def exp(self):
        """Exponential. Track in the graph."""
        raise NotImplementedError

    def backward(self):
        """
        Run backpropagation from this Value.

        Steps:
        1. Build a topological ordering of the graph (all ancestors of this node)
        2. Set this node's gradient to 1.0 (dself/dself = 1)
        3. Walk the topo order in reverse, calling _backward on each node
        """
        raise NotImplementedError
