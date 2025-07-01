import torch
from exercise import Exercise

class CovarianceMatrix(Exercise):
    """
    Build the 2×2 covariance matrix for two paired arrays.
    Covariance formula: cov(X,Y) = Σ((xi - μx)(yi - μy)) / N
    
    Example:
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        y = torch.tensor([2.0, 4.0, 5.0, 4.0, 5.0])
        result = solve(x, y)  # Returns: [[2.0, 0.8], [0.8, 1.2]]
    """
    def solve(self, x, y):
        pass

"""TESTS: DO NOT MODIFY"""
def test_covariance_basic():
    """Test basic covariance matrix calculation"""
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    y = torch.tensor([2.0, 4.0, 5.0, 4.0, 5.0])
    result = CovarianceMatrix().solve(x, y)
    # Check diagonal elements (variances)
    assert abs(result[0, 0].item() - 2.0) < 1e-6
    # Check off-diagonal (covariances are symmetric)
    assert abs(result[0, 1].item() - result[1, 0].item()) < 1e-6


def test_covariance_perfect_correlation():
    """Test covariance with perfectly correlated data"""
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    y = 2 * x + 1  # Perfect linear relationship
    result = CovarianceMatrix().solve(x, y)
    # Covariance should be 2 * var(x)
    assert abs(result[0, 1].item() - 4.0) < 1e-6


def test_covariance_uncorrelated():
    """Test covariance with uncorrelated data"""
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    y = torch.tensor([2.0, 2.0, 2.0, 2.0])  # Constant y
    result = CovarianceMatrix().solve(x, y)
    # Covariance should be zero
    assert abs(result[0, 1].item()) < 1e-6
    assert abs(result[1, 0].item()) < 1e-6


def test_covariance_negative_correlation():
    """Test covariance with negative correlation"""
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    y = -x + 6  # Negative linear relationship
    result = CovarianceMatrix().solve(x, y)
    # Covariance should be negative
    assert result[0, 1].item() < 0


def test_covariance_symmetry():
    """Test that covariance matrix is symmetric"""
    x = torch.randn(10)
    y = torch.randn(10)
    result = CovarianceMatrix().solve(x, y)
    assert torch.allclose(result, result.T)


def test_covariance_empty():
    """Test covariance with empty tensors"""
    x = torch.tensor([])
    y = torch.tensor([])
    result = CovarianceMatrix().solve(x, y)
    assert torch.all(torch.isnan(result))


def test_covariance_single_point():
    """Test covariance with single data point"""
    x = torch.tensor([1.0])
    y = torch.tensor([2.0])
    result = CovarianceMatrix().solve(x, y)
    # All values should be zero (no variance)
    assert torch.allclose(result, torch.zeros(2, 2))
