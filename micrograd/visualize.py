from graphviz import Digraph


def draw_graph(root):
    """
    Visualize a Value's computation graph using graphviz.

    Each node shows:
    - the data value
    - the gradient
    - the operation that created it (if any)

    Returns a Digraph object (call .render() or display in a notebook).

    Hint: walk the graph from root, collecting all nodes and edges.
    """
    raise NotImplementedError
