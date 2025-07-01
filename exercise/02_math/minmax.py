import torch
from exercise import Exercise

class MinMaxScale(Exercise):
    """
    Transform a tensor into the [0,1] range using min-max scaling.
    Formula: (x - min(x)) / (max(x) - min(x))
    
    Example:
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        result = solve(x)  # Returns: torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    """
    def solve(self, x):
        pass

"""TESTS: DO NOT MODIFY"""
def test_minmax_basic():
    """Test basic min-max scaling"""
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    result = MinMaxScale().solve(x)
    expected = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    assert torch.allclose(result, expected)


def test_minmax_already_normalized():
    """Test scaling already normalized data"""
    x = torch.tensor([0.0, 0.5, 1.0])
    result = MinMaxScale().solve(x)
    assert torch.allclose(result, x)


def test_minmax_negative_values():
    """Test scaling with negative values"""
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    result = MinMaxScale().solve(x)
    expected = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    assert torch.allclose(result, expected)


def test_minmax_constant_values():
    """Test scaling when all values are the same"""
    x = torch.full((5,), 3.14)
    result = MinMaxScale().solve(x)
    assert torch.all(result == 0.0)


def test_minmax_two_values():
    """Test scaling with only two values"""
    x = torch.tensor([10.0, 20.0])
    result = MinMaxScale().solve(x)
    assert torch.allclose(result, torch.tensor([0.0, 1.0]))


def test_minmax_preserves_order():
    """Test that scaling preserves order"""
    x = torch.tensor([5.0, 2.0, 8.0, 1.0, 9.0])
    result = MinMaxScale().solve(x)
    # Check min and max
    assert result.min() == 0.0
    assert result.max() == 1.0
    # Check order is preserved
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            if x[i] < x[j]:
                assert result[i] < result[j]
            elif x[i] > x[j]:
                assert result[i] > result[j]


def test_minmax_dtype():
    """Test that dtype is preserved"""
    x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    result = MinMaxScale().solve(x)
    assert result.dtype == torch.float64