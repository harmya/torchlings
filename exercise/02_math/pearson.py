import torch
from exercise import Exercise

class PearsonCorrelation(Exercise):
    """
    Implement Pearson's correlation coefficient (r) for two 1D tensors.
    Formula: r = cov(X,Y) / (σx * σy)
    
    Example:
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        y = torch.tensor([2.0, 4.0, 6.0, 8.0, 10.0])  # Perfect correlation
        result = solve(x, y)  # Returns: 1.0
    """
    def solve(self, x, y):
        pass

"""TESTS: DO NOT MODIFY"""
def test_pearson_perfect_positive():
    """Test perfect positive correlation"""
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    y = 2 * x + 3
    result = PearsonCorrelation().solve(x, y)
    assert abs(result.item() - 1.0) < 1e-6


def test_pearson_perfect_negative():
    """Test perfect negative correlation"""
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    y = -x + 10
    result = PearsonCorrelation().solve(x, y)
    assert abs(result.item() - (-1.0)) < 1e-6


def test_pearson_no_correlation():
    """Test no correlation"""
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    y = torch.tensor([3.0, 3.0, 3.0, 3.0, 3.0])  # Constant
    result = PearsonCorrelation().solve(x, y)
    assert torch.isnan(result)  # Undefined when one variable is constant


def test_pearson_moderate_correlation():
    """Test moderate correlation"""
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    y = torch.tensor([2.0, 4.0, 5.0, 4.0, 5.0])
    result = PearsonCorrelation().solve(x, y)
    assert 0 < result.item() < 1  # Should be positive but not perfect


def test_pearson_single_point():
    """Test with single data point"""
    x = torch.tensor([1.0])
    y = torch.tensor([2.0])
    result = PearsonCorrelation().solve(x, y)
    assert torch.isnan(result)


def test_pearson_two_points():
    """Test with two data points"""
    x = torch.tensor([1.0, 2.0])
    y = torch.tensor([3.0, 5.0])
    result = PearsonCorrelation().solve(x, y)
    # With only 2 points, correlation is either 1, -1, or undefined
    assert abs(abs(result.item()) - 1.0) < 1e-6


def test_pearson_dtype_preserved():
    """Test that computation works with different dtypes"""
    x = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    y = torch.tensor([2.0, 4.0, 6.0, 8.0], dtype=torch.float64)
    result = PearsonCorrelation().solve(x, y)
    assert result.dtype == torch.float64
