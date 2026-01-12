# The chain rule in autograd
import torch


def chain_rule_example():
    # TODO: Create a scalar tensor with value 2.0 and requires_grad=True
    x = None

    # TODO: Compute y = 3x^2 + 2x + 1
    y = None

    # TODO: Call backward() on y, then return x.grad
    # The derivative should be 6x + 2 = 14 when x = 2
    return None


"""
----------------------TESTS-------------------------
------------------DO NOT TOUCH TESTS----------------
"""


def test_chain_rule():
    grad = chain_rule_example()
    assert grad.item() == 14.0
