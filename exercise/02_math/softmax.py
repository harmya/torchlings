import torch
from exercise import Exercise

class Softmax(Exercise):
    """
    Compute softmax across a vector with numeric stability.
    Formula: softmax(xi) = exp(xi - max(x)) / Σexp(xj - max(x))
    
    Example:
        x = torch.tensor([1.0, 2.0, 3.0])
        result = solve(x)  # Returns normalized probabilities that sum to 1
    """
    def solve(self, x):
        pass

"""TESTS: DO NOT MODIFY"""
def test_softmax_basic():
    """Test basic softmax calculation"""
    x = torch.tensor([1.0, 2.0, 3.0])
    result = Softmax().solve(x)
    # Check that it sums to 1
    assert abs(result.sum().item() - 1.0) < 1e-6
    # Check that ordering is preserved
    assert result[0] < result[1] < result[2]


def test_softmax_uniform():
    """Test softmax with uniform values"""
    x = torch.ones(4)
    result = Softmax().solve(x)
    # All probabilities should be equal
    assert torch.allclose(result, torch.full((4,), 0.25))


def test_softmax_large_values():
    """Test numerical stability with large values"""
    x = torch.tensor([1000.0, 1001.0, 1002.0])
    result = Softmax().solve(x)
    # Should not overflow
    assert torch.all(torch.isfinite(result))
    assert abs(result.sum().item() - 1.0) < 1e-6


def test_softmax_negative_values():
    """Test softmax with negative values"""
    x = torch.tensor([-1.0, -2.0, -3.0])
    result = Softmax().solve(x)
    assert abs(result.sum().item() - 1.0) < 1e-6
    # Ordering should still be preserved
    assert result[0] > result[1] > result[2]


def test_softmax_single_element():
    """Test softmax with single element"""
    x = torch.tensor([5.0])
    result = Softmax().solve(x)
    assert result.item() == 1.0


def test_softmax_extreme_values():
    """Test softmax with extreme value differences"""
    x = torch.tensor([0.0, 100.0])
    result = Softmax().solve(x)
    # Second element should dominate
    assert result[1] > 0.99
    assert result[0] < 0.01
    assert abs(result.sum().item() - 1.0) < 1e-6


def test_softmax_dtype():
    """Test softmax preserves dtype"""
    x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    result = Softmax().solve(x)
    assert result.dtype == torch.float64
