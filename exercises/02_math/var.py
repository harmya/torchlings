import torch
from exercise import Exercise

class ManualVariance(Exercise):
    """
    Calculate variance (σ²) from raw data without using .var().
    Uses the formula: σ² = Σ(xi - μ)² / N for population variance.
    
    Example:
        data = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        result = solve(data)  # Returns: 2.0
        # Mean = 3.0, variance = ((1-3)² + (2-3)² + (3-3)² + (4-3)² + (5-3)²) / 5 = 2.0
    """
    def solve(self, data):
        pass

"""TESTS: DO NOT MODIFY"""
def test_variance_basic():
    """Test basic variance calculation"""
    data = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ManualVariance().solve(data)
    assert abs(result.item() - 2.0) < 1e-6


def test_variance_constant():
    """Test variance of constant values is zero"""
    data = torch.full((5,), 7.0)
    result = ManualVariance().solve(data)
    assert abs(result.item()) < 1e-6


def test_variance_two_values():
    """Test variance with two values"""
    data = torch.tensor([1.0, 5.0])
    result = ManualVariance().solve(data)
    # Mean = 3.0, variance = ((1-3)² + (5-3)²) / 2 = (4 + 4) / 2 = 4.0
    assert abs(result.item() - 4.0) < 1e-6


def test_variance_negative_values():
    """Test variance with negative values"""
    data = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    result = ManualVariance().solve(data)
    assert abs(result.item() - 2.0) < 1e-6


def test_variance_single_value():
    """Test variance of single value is zero"""
    data = torch.tensor([42.0])
    result = ManualVariance().solve(data)
    assert abs(result.item()) < 1e-6


def test_variance_empty():
    """Test variance of empty tensor returns nan"""
    data = torch.tensor([])
    result = ManualVariance().solve(data)
    assert torch.isnan(result)


def test_variance_dtype_preserved():
    """Test that dtype is preserved"""
    data = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    result = ManualVariance().solve(data)
    assert result.dtype == torch.float64
