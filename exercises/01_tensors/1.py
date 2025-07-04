import torch

# Tensors are a fundamental data structure in PyTorch.
# Think of them as a multi-dimensional numpy array which can be used on GPUs.


def create_tensor():
    # TODO: Create a tensor with values 1, 2, 3, 4, 5
    result = None
    return result


def create_tensor_from_array(array):
    # TODO: Create a tensor from a given list of numbers
    result = None
    return result


"""
----------------------TESTS---------------------
---------------DO NOT TOUCH TESTS---------------
"""


def test_tensor():
    tensor = create_tensor()
    assert tensor is not None
    assert tensor.tolist() == [1, 2, 3, 4, 5]
    assert tensor.shape == torch.Size([5])


def test_create_from_list():
    arr = [2, 4, 6, 8]
    tensor = create_tensor_from_array(arr)
    assert tensor is not None
    assert tensor.tolist() == arr
    assert tensor.shape == torch.Size([len(arr)])
