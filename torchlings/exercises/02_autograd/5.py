# Controlling which parts of computation contribute to gradients
import torch


def selective_gradient_branches():
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

    # Branch 1: Full gradient flow
    y1 = x**2

    # TODO: Branch 2: Create a modified copy of x where x[1] is detached
    # so it doesn't contribute to gradients through this branch
    x_modified = x.clone()
    x_modified[1] = None  # detach the middle element
    y2 = x_modified**3

    # TODO: Branch 3: Use torch.no_grad() to compute temp = x * 2
    # without tracking, then add x (which IS tracked) to get y3
    temp = None
    y3 = None

    # Combine all branches
    total = y1.sum() + y2.sum() + y3.sum()

    # TODO: Compute gradients
    total.backward()

    # Expected gradients:
    # x[0]: 2*1 (from y1) + 3*1^2 (from y2) + 1 (from y3) = 6
    # x[1]: 2*2 (from y1) + 0 (detached in y2) + 1 (from y3) = 5
    # x[2]: 2*3 (from y1) + 3*3^2 (from y2) + 1 (from y3) = 34

    return x.grad


"""
----------------------TESTS-------------------------
------------------DO NOT TOUCH TESTS----------------
"""


def test_selective_gradients():
    grad = selective_gradient_branches()
    expected = torch.tensor([6.0, 5.0, 34.0])
    assert torch.allclose(grad, expected)
