import torch
from exercise import Exercise
import math

class L2Norm(Exercise):
    """
    Compute L2 norm (Euclidean norm) of a vector: √(Σxi²)
    
    Example:
        x = torch.tensor([3.0, 4.0])
        result = solve(x)  # Returns: 5.0 (√(9+16) = √25 = 5)
    """
    def solve(self, x):
        pass

"""TESTS: DO NOT MODIFY"""
def test_l2norm_basic():
    """Test basic L2 norm calculation"""
    x = torch.tensor([3.0, 4.0])
    result = L2Norm().solve(x)
    assert abs(result.item() - 5.0) < 1e-6


def test_l2norm_unit_vector():
    """Test L2 norm of unit vector"""
    x = torch.tensor([1.0, 0.0, 0.0])
    result = L2Norm().solve(x)
    assert abs(result.item() - 1.0) < 1e-6


def test_l2norm_zero_vector():
    """Test L2 norm of zero vector"""
    x = torch.zeros(5)
    result = L2Norm().solve(x)
    assert result.item() == 0.0


def test_l2norm_negative_values():
    """Test L2 norm with negative values"""
    x = torch.tensor([-3.0, -4.0])
    result = L2Norm().solve(x)
    assert abs(result.item() - 5.0) < 1e-6


def test_l2norm_single_element():
    """Test L2 norm of single element vector"""
    x = torch.tensor([-7.0])
    result = L2Norm().solve(x)
    assert abs(result.item() - 7.0) < 1e-6


def test_l2norm_high_dimensional():
    """Test L2 norm in higher dimensions"""
    x = torch.ones(10)
    result = L2Norm().solve(x)
    assert abs(result.item() - math.sqrt(10)) < 1e-6


def test_l2norm_dtype_preserved():
    """Test dtype handling"""
    x = torch.tensor([3.0, 4.0], dtype=torch.float64)
    result = L2Norm().solve(x)
    assert result.dtype == torch.float64
