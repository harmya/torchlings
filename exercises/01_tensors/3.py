import torch


def add_tensors():
    a = torch.tensor([1, 2, 3])
    b = torch.tensor([4, 5, 6])
    # TODO: Add tensors a and b together

    result = None
    return result


def multiply_scalar():
    tensor = torch.tensor([1, 2, 3, 4])
    scalar = 5
    # TODO: Multiply the tensor by the scalar

    result = None
    return result


def matrix_multiply():
    a = torch.tensor([[1, 2], [3, 4]])
    b = torch.tensor([[5, 6], [7, 8]])
    # TODO: Perform matrix multiplication between a and b

    result = None
    return result


"""
----------------------TESTS---------------------
---------------DO NOT TOUCH TESTS---------------
"""


def test_add():
    assert add_tensors().tolist() == [5, 7, 9]


def test_scalar_mult():
    assert multiply_scalar().tolist() == [5, 10, 15, 20]


def test_mm():
    assert matrix_multiply().tolist() == [[19, 22], [43, 50]]
