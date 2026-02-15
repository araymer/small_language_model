"""
Verify micrograd gradients against PyTorch.

Build the same expression in both systems, run backward,
and compare the gradients to make sure they match.
"""
import torch
from micrograd.engine import Value


def verify():
    # -- micrograd --
    a = Value(2.0)
    b = Value(-3.0)
    c = Value(0.5)

    d = a * b           # -6.0
    e = d + c           # -5.5
    f = e ** 2           # 30.25
    g = f.tanh()         # ~1.0 (tanh saturates for large inputs)
    g.backward()

    # -- pytorch --
    at = torch.tensor(2.0, requires_grad=True)
    bt = torch.tensor(-3.0, requires_grad=True)
    ct = torch.tensor(0.5, requires_grad=True)

    dt = at * bt
    et = dt + ct
    ft = et ** 2
    gt = ft.tanh()
    gt.backward()

    # -- compare --
    tol = 1e-6
    pairs = [('a', a.grad, at.grad.item()),
             ('b', b.grad, bt.grad.item()),
             ('c', c.grad, ct.grad.item())]

    for name, mg, pt in pairs:
        diff = abs(mg - pt)
        status = "PASS" if diff < tol else "FAIL"
        print(f"  {name}: micrograd={mg:.8f}  pytorch={pt:.8f}  diff={diff:.2e}  [{status}]")
        assert diff < tol, f"Gradient mismatch for {name}: {mg} vs {pt}"


if __name__ == '__main__':
    verify()
    print("\nAll gradients match!")
