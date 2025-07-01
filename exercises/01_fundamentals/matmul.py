import torch
import pytest
from exercise import Exercise

class MatMatMul(Exercise):
    """
    Perform matrix multiplication of two 2D matrices.
    
    Example:
        a = torch.tensor([[1, 2],
                         [3, 4]])  # Shape: (2, 2)
        b = torch.tensor([[5, 6],
                         [7, 8]])  # Shape: (2, 2)
        result = solve(a, b)  # Returns: torch.tensor([[19, 22], [43, 50]])
    """
    def solve(self, a, b):
        pass

"""TESTS: DO NOT MODIFY"""
def test_matmat_basic():
    """Test basic matrix multiplication with identity matrix"""
    a = torch.tensor([[1,0],[0,1]], dtype=torch.float32)
    b = torch.tensor([[5,6],[7,8]], dtype=torch.float32)
    out = MatMatMul().solve(a, b)
    assert torch.equal(out, b)

def test_matmat_shape():
    """Test output shape for matrix multiplication"""
    a = torch.zeros(3,2)
    b = torch.zeros(2,4)
    out = MatMatMul().solve(a,b)
    assert out.shape == (3,4)

def test_matmat_dtype():
    """Test that dtype is preserved"""
    a = torch.ones(2,2, dtype=torch.int32)
    b = torch.full((2,2,), 3, dtype=torch.int32)
    out = MatMatMul().solve(a,b)
    assert out.dtype == torch.int32

def test_matmat_mismatch_error():
    """Test that incompatible shapes raise error"""
    with pytest.raises(RuntimeError):
        MatMatMul().solve(torch.zeros(2,3), torch.zeros(3,2,2))

def test_matmat_random():
    """Test specific element calculation"""
    a = torch.randn(4,4)
    b = torch.randn(4,4)
    out = MatMatMul().solve(a,b)
    expected = sum(a[2,i]*b[i,3] for i in range(4))
    assert abs(out[2,3].item() - expected.item()) < 1e-6

def test_matmat_rectangular():
    """Test multiplication of rectangular matrices"""
    a = torch.tensor([[1, 2, 3]], dtype=torch.float32)  # 1x3
    b = torch.tensor([[4], [5], [6]], dtype=torch.float32)  # 3x1
    out = MatMatMul().solve(a, b)
    assert out.shape == (1, 1)
    assert out.item() == 32.0  # 1*4 + 2*5 + 3*6

def test_matmat_zero_matrix():
    """Test multiplication with zero matrix"""
    a = torch.randn(3, 3)
    b = torch.zeros(3, 3)
    out = MatMatMul().solve(a, b)
    assert torch.all(out == 0)