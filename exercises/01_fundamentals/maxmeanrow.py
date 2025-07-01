import torch
import pytest
from exercise import Exercise

class MaxMeanRow(Exercise):
    """
    Find the index of the row with the maximum mean value in a 2D matrix.
    If multiple rows have the same maximum mean, return the first one.
    
    Example:
        mat = torch.tensor([[1, 2],      # mean = 1.5
                           [5, 0],       # mean = 2.5
                           [3, 3]])      # mean = 3.0
        result = solve(mat)  # Returns: 2 (index of row with highest mean)
    """
    def solve(self, mat):
        pass

"""TESTS: DO NOT MODIFY"""
def test_max_mean_basic():
    """Test finding row with maximum mean"""
    mat = torch.tensor([[1,2],[5,0],[3,3]])
    idx = MaxMeanRow().solve(mat)
    # means = [1.5, 2.5, 3.0] ⇒ row 2
    assert idx.item() == 2

def test_max_mean_tie():
    """Test that ties return the first index"""
    mat = torch.tensor([[1,1],[2,0]])
    idx = MaxMeanRow().solve(mat)
    # both rows mean=1 ⇒ returns first (0)
    assert idx.item() == 0

def test_max_mean_dtype():
    """Test return type is integer"""
    mat = torch.randn(4,4, dtype=torch.float32)
    idx = MaxMeanRow().solve(mat)
    assert isinstance(idx.item(), int)

def test_max_mean_single_row():
    """Test with single row"""
    mat = torch.tensor([[42]])
    idx = MaxMeanRow().solve(mat)
    assert idx.item() == 0

def test_max_mean_empty_error():
    """Test that empty matrix raises error"""
    with pytest.raises(IndexError):
        MaxMeanRow().solve(torch.zeros(0,3))

def test_max_mean_negative_values():
    """Test with negative values"""
    mat = torch.tensor([[-5, -3],   # mean = -4
                        [-1, -1],    # mean = -1
                        [-2, -4]])   # mean = -3
    idx = MaxMeanRow().solve(mat)
    assert idx.item() == 1  # Row with mean -1 is the maximum

def test_max_mean_integer_input():
    """Test with integer input (should convert to float for mean)"""
    mat = torch.tensor([[1, 2],
                        [3, 4],
                        [5, 6]], dtype=torch.int32)
    idx = MaxMeanRow().solve(mat)
    # means = [1.5, 3.5, 5.5] ⇒ row 2
    assert idx.item() == 2
