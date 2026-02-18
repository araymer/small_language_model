from engine import Value
from nn import MLP

xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
ys = [1.0, -1.0, -1.0, 1.0]

train_set = list(zip(xs, ys))


def train(iterations=100, learning_rate=0.05):
    model = MLP(3, [4, 4, 1])

    for i in range(iterations):
        loss = 0
        for x, y in train_set:
            x, y = [Value(xi) for xi in x], Value(y)
            pred = model(x)
            loss += (pred - y) ** 2
        for p in model.parameters():
            p.grad = 0
        loss.backward()
        for p in model.parameters():
            p.data -= p.grad * learning_rate
        print(f"iteration {i}: loss = {loss.data:.6f}")


if __name__ == '__main__':
    train()
