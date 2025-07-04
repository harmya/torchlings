# Advanced tensor indexing, slicing, and gathering operations

import torch


def advanced_indexing():
    tensor = torch.arange(20).reshape(4, 5)
    # TODO: Use advanced indexing to get elements at positions (0,1), (1,2), (2,3), (3,4)
    # Hint: Use two lists/tensors for row and column indices

    result = None
    return result


def gather_operation():
    tensor = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    indices = torch.tensor([[0, 2], [1, 0], [2, 1]])
    # TODO: Collect elements along dim=1
    # Bonus points if you use a inbuilt torch function
    # Expected: [[1, 3], [5, 4], [9, 8]]

    result = None
    return result


def masked_selection():
    tensor = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # TODO: Select all elements greater than 5 using masked_select

    result = None
    return result


def index_add_operation():
    tensor = torch.ones(5, 3)
    indices = torch.tensor([0, 2, 4])
    values = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
    # TODO: Add values to tensor at specified row indices

    result = None
    return result


def narrow_and_select():
    tensor = torch.arange(30).reshape(5, 6)
    # TODO: Narrow the tensor to get rows 1-3 (exclusive end) and columns 2-5
    # TODO: Then select the diagonal of this narrowed tensor

    narrowed, diag = None, None
    return narrowed, diag


"""
----------------------TESTS---------------------
---------------DO NOT TOUCH TESTS---------------
"""


def test_advanced_indexing():
    assert advanced_indexing().tolist() == [1, 7, 13, 19]


def test_gather_operation():
    assert gather_operation().tolist() == [[1, 3], [5, 4], [9, 8]]


def test_masked_selection():
    assert masked_selection().tolist() == [6, 7, 8, 9]


def test_index_add_operation():
    result = index_add_operation()
    assert result[0, 0] == 2.0  # 1.0 + 1.0
    assert result[2, 1] == 6.0  # 1.0 + 5.0


def test_narrow_and_select():
    narrowed, diag = narrow_and_select()
    assert narrowed.shape == torch.Size([2, 3])
    assert diag.tolist() == [8, 15]
