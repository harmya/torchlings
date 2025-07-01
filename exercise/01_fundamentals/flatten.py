import torch
from exercise import Exercise

class Flatten1D(Exercise):
    """
    Flatten a tensor of any shape into a 1D vector.
    
    Example:
        x = torch.tensor([[1, 2, 3],
                         [4, 5, 6]])  # Shape: (2, 3)
        result = solve(x)  # Returns: torch.tensor([1, 2, 3, 4, 5, 6])  # Shape: (6,)
    """
    def solve(self, x):
        pass

"""TESTS: DO NOT MODIFY"""
def test_flatten_2d():
    """Test flattening a 2D tensor"""
    x = torch.arange(6).view(2,3)
    out = Flatten1D().solve(x)
    assert out.shape == (6,)
    assert torch.equal(out, torch.tensor([0,1,2,3,4,5]))

def test_flatten_3d():
    """Test flattening a 3D tensor"""
    x = torch.ones(2,2,2)
    out = Flatten1D().solve(x)
    assert out.numel() == 8
    assert out.shape == (8,)

def test_flatten_empty():
    """Test flattening an empty tensor"""
    x = torch.zeros(0,5)
    out = Flatten1D().solve(x)
    assert out.numel() == 0 and out.shape == (0,)

def test_flatten_dtype_preserved():
    """Test that dtype is preserved"""
    x = torch.randn(4,4, dtype=torch.float64)
    out = Flatten1D().solve(x)
    assert out.dtype == x.dtype

def test_flatten_content():
    """Test flattening preserves content"""
    x = torch.tensor([[5]])
    out = Flatten1D().solve(x)
    assert out.tolist() == [5]

def test_flatten_4d():
    """Test flattening a 4D tensor"""
    x = torch.arange(24).view(2, 3, 2, 2)
    out = Flatten1D().solve(x)
    assert out.shape == (24,)
    assert torch.equal(out, torch.arange(24))

def test_flatten_already_1d():
    """Test flattening a tensor that's already 1D"""
    x = torch.tensor([1, 2, 3, 4, 5])
    out = Flatten1D().solve(x)
    assert out.shape == (5,)
    assert torch.equal(out, x)
