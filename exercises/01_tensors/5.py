import torch


def reshape_tensor():
    tensor = torch.arange(12)
    # TODO: Reshape to 3x4

    result = None
    return result


def add_dimension():
    tensor = torch.tensor([1, 2, 3, 4])
    # TODO: Add a new dimension to make it 4x1

    result = None
    return result


def remove_dim_tensor():
    tensor = torch.zeros(1, 3, 1, 4)
    # TODO: Remove dimensions of size 1

    result = None
    return result


def transpose_tensor():
    tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
    # TODO: Transpose the tensor

    result = None
    return result


"""
----------------------TESTS---------------------
---------------DO NOT TOUCH TESTS---------------
"""


def test_reshaping():
    assert reshape_tensor().shape == torch.Size([3, 4])


def test_add_dim():
    assert add_dimension().shape == torch.Size([4, 1])


def test_remove_dim():
    assert remove_dim_tensor().shape == torch.Size([3, 4])


def test_transpose():
    assert transpose_tensor().shape == torch.Size([3, 2])
