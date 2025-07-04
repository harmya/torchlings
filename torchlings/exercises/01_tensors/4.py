import torch


def get_element():
    tensor = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # TODO: Get the element at row 1, column 2 (should be 6)

    result = None
    return result


def get_row():
    tensor = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # TODO: Get the second row (index 1)

    result = None
    return result


def get_column():
    tensor = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # TODO: Get the third column (index 2)

    result = None
    return result


def slice_tensor():
    tensor = torch.arange(20).reshape(4, 5)
    # TODO: Get elements from rows 1-2 (inclusive) and columns 2-4 (exclusive)

    result = None
    return result


"""
----------------------TESTS---------------------
---------------DO NOT TOUCH TESTS---------------
"""


def test_indexing():
    assert get_element().item() == 6


def test_get_row():
    assert get_row().tolist() == [4, 5, 6]


def test_get_col():
    assert get_column().tolist() == [3, 6, 9]


def test_get_slice():
    assert slice_tensor().tolist() == [[7, 8, 9], [12, 13, 14]]
