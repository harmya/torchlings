import torch
import pytest
from exercise import Exercise

class PermuteAxes(Exercise):
    """
    Swap two axes (dimensions) of a tensor.
    
    Example:
        x = torch.tensor([[1, 2, 3],
                         [4, 5, 6]])  # Shape: (2, 3)
        result = solve(x, 0, 1)  # Swap axis 0 and 1
        # Returns: torch.tensor([[1, 4],
        #                        [2, 5],
        #                        [3, 6]])  # Shape: (3, 2)
    """
    def solve(self, x, axis1, axis2):
        pass

"""TESTS: DO NOT MODIFY"""
def test_permute_simple():
    """Test simple 2D transpose"""
    x = torch.arange(6).view(2,3)
    out = PermuteAxes().solve(x, 0, 1)
    assert out.shape == (3,2)
    assert torch.equal(out, torch.tensor([[0,3],[1,4],[2,5]]))

def test_permute_3d():
    """Test transposing axes in 3D tensor"""
    x = torch.zeros(2,3,4)
    out = PermuteAxes().solve(x, 1, 2)
    assert out.shape == (2,4,3)

def test_permute_dtype():
    """Test that dtype is preserved"""
    x = torch.ones(1,1, dtype=torch.int32)
    out = PermuteAxes().solve(x, 0, 1)
    assert out.dtype == x.dtype

def test_permute_invalid_axis():
    """Test that invalid axis raises error"""
    with pytest.raises(IndexError):
        PermuteAxes().solve(torch.zeros(2,2), 0, 2)

def test_permute_content():
    """Test that content is correctly permuted"""
    x = torch.tensor([[[1,2],[3,4]]])  # shape (1,2,2)
    out = PermuteAxes().solve(x, 1, 2)
    assert out.shape == (1,2,2)
    assert out[0,1,0].item() == 2

def test_permute_same_axis():
    """Test swapping an axis with itself (no-op)"""
    x = torch.randn(3, 4, 5)
    out = PermuteAxes().solve(x, 1, 1)
    assert torch.equal(out, x)

def test_permute_negative_indices():
    """Test using negative indices"""
    x = torch.arange(24).view(2, 3, 4)
    out1 = PermuteAxes().solve(x, 0, -1)  # Swap first and last axis
    out2 = PermuteAxes().solve(x, 0, 2)   # Should be the same
    assert torch.equal(out1, out2)
    assert out1.shape == (4, 3, 2)