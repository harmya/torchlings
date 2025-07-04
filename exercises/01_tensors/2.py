# Creating tensors with specific shapes and values

import torch


def create_zeros_tensor():
    # TODO: Create a 3x4 tensor filled with zeros
    result = None
    return result


def create_ones_tensor():
    # TODO: Create a 2x2x2 tensor filled with ones
    result = None
    return result


def create_random_tensor():
    # TODO: Create a 5x5 tensor with random values between 0 and 1
    result = None
    return result


"""
----------------------TESTS---------------------
---------------DO NOT TOUCH TESTS---------------
"""


def test_zeros():
    zeros = create_zeros_tensor()
    assert zeros.shape == torch.Size([3, 4])
    assert torch.all(zeros == 0)


def test_ones():
    ones = create_ones_tensor()
    assert ones.shape == torch.Size([2, 2, 2])
    assert torch.all(ones == 1)


def test_random():
    random = create_random_tensor()
    assert random.shape == torch.Size([5, 5])
    assert torch.all((random >= 0) & (random <= 1))
