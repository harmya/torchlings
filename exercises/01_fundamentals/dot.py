import torch
import pytest
from exercise import Exercise

class VectorDot(Exercise):
    """
    Compute the dot product of two 1D vectors.
    
    Example:
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([4.0, 5.0, 6.0])
        result = solve(a, b)  # Returns: 1*4 + 2*5 + 3*6 = 32.0
    """
    def solve(self, a, b):
        pass

"""TESTS: DO NOT MODIFY"""
def test_dot_basic():
    """Test basic dot product calculation"""
    a = torch.tensor([1.,2.,3.])
    b = torch.tensor([4.,5.,6.])
    assert VectorDot().solve(a,b).item() == 32.0

def test_dot_zero_length():
    """Test dot product of empty vectors"""
    a = torch.empty(0)
    b = torch.empty(0)
    assert VectorDot().solve(a,b).item() == 0.0

def test_dot_dtype():
    """Test that dtype is preserved"""
    a = torch.ones(4, dtype=torch.int32)
    b = torch.full((4,), 2, dtype=torch.int32)
    out = VectorDot().solve(a,b)
    assert out.dtype == torch.int32
    assert out.item() == 8

def test_dot_mismatch_error():
    """Test that mismatched vector lengths raise error"""
    with pytest.raises(RuntimeError):
        VectorDot().solve(torch.ones(3), torch.ones(4))

def test_dot_negative_values():
    """Test dot product with negative values"""
    a = torch.tensor([-1., 1.])
    b = torch.tensor([1., -1.])
    assert VectorDot().solve(a,b).item() == -2.0

def test_dot_orthogonal_vectors():
    """Test dot product of orthogonal vectors"""
    a = torch.tensor([1., 0., 0.])
    b = torch.tensor([0., 1., 0.])
    assert VectorDot().solve(a,b).item() == 0.0

def test_dot_unit_vectors():
    """Test dot product of parallel unit vectors"""
    a = torch.tensor([0.6, 0.8])
    b = torch.tensor([0.6, 0.8])
    assert abs(VectorDot().solve(a,b).item() - 1.0) < 1e-6

