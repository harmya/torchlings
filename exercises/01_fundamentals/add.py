import torch
from exercise import Exercise

# Exercise 1: Add
class Add(Exercise):
    """
    Implement element-wise addition of two tensors.
    
    Example:
        a = torch.tensor([1, 2, 3])
        b = torch.tensor([4, 5, 6])
        result = solve(a, b)  # Returns: torch.tensor([5, 7, 9])
    """
    
    def solve(self, a, b):
        pass

"""TESTS: DO NOT MODIFY"""
def test_add():
    """Test that the add function returns the correct sum for scalars"""
    assert Add().solve(torch.tensor(5), torch.tensor(5)).item() == 10

def test_add_shape():
    """Test that the add function returns a tensor of the correct shape"""
    a = torch.ones(5)
    b = torch.ones(5)
    result = Add().solve(a, b)
    assert result.shape == (5,)

def test_add_type():
    """Test that the add function returns a tensor of the correct type"""
    a = torch.ones(5, dtype=torch.float32)
    b = torch.ones(5, dtype=torch.float32)
    result = Add().solve(a, b)
    assert result.dtype == torch.float32

def test_add_values():
    """Test that all values in the tensor are the sum of the inputs"""
    a = torch.full((10,), 2.0)
    b = torch.full((10,), 3.0)
    result = Add().solve(a, b)
    assert torch.all(result == 5.0)
    assert result.sum().item() == 50.0

def test_add_zero_size():
    """Test adding tensors with size 0"""
    a = torch.ones(0)
    b = torch.ones(0)
    result = Add().solve(a, b)
    assert result.shape == (0,)
    assert result.numel() == 0

def test_add_large_size():
    """Test adding large tensors"""
    a = torch.ones(1000)
    b = torch.ones(1000)
    result = Add().solve(a, b)
    assert result.shape == (1000,)
    assert torch.all(result == 2.0)

def test_add_broadcasting():
    """Test broadcasting addition"""
    a = torch.ones(3, 1)
    b = torch.ones(1, 4)
    result = Add().solve(a, b)
    assert result.shape == (3, 4)
    assert torch.all(result == 2.0)

def test_add_negative_values():
    """Test addition with negative values"""
    a = torch.tensor([-1.0, -2.0, -3.0])
    b = torch.tensor([1.0, 2.0, 3.0])
    result = Add().solve(a, b)
    assert torch.all(result == 0.0)