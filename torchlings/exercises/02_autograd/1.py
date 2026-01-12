# Computing gradients with autograd
import torch


def simple_gradient():
    # TODO: Create a tensor with requires_grad=True
    x = torch.tensor([2.0, 3.0, 4.0])

    # TODO: Compute y = x^2
    y = None

    # TODO: Compute the mean of y
    z = None

    # TODO: Call backward() on z, then return x.grad
    return None


"""
----------------------TESTS-------------------------
------------------DO NOT TOUCH TESTS----------------
"""


def test_gradient():
    grad = simple_gradient()
    expected = torch.tensor([4 / 3, 2.0, 8 / 3])
    assert torch.allclose(grad, expected, atol=1e-5)
