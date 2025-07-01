import torch
from exercise import Exercise

class ColwiseMean(Exercise):
    """
    Compute the column-wise mean of a 2D tensor.
    Returns a 1D tensor where each element is the mean of the corresponding column.
    
    Example:
        data = torch.tensor([[1.0, 2.0, 3.0],
                            [4.0, 5.0, 6.0],
                            [7.0, 8.0, 9.0]])
        result = solve(data)  # Returns: torch.tensor([4.0, 5.0, 6.0])
        # Column 0: (1+4+7)/3 = 4.0
        # Column 1: (2+5+8)/3 = 5.0  
        # Column 2: (3+6+9)/3 = 6.0
    """
    def solve(self, data):
        pass

"""TESTS: DO NOT MODIFY"""

def test_mean_basic():
    """Test basic column-wise mean calculation"""
    data = torch.tensor([[1.0, 2.0, 3.0],
                         [4.0, 5.0, 6.0],
                         [7.0, 8.0, 9.0]])
    result = ColwiseMean().solve(data)
    expected = torch.tensor([4.0, 5.0, 6.0])
    assert torch.allclose(result, expected)


def test_mean_single_row():
    """Test mean of single row (returns the row itself)"""
    data = torch.tensor([[42.0, 3.14, 2.718]])
    result = ColwiseMean().solve(data)
    assert torch.allclose(result, data[0])


def test_mean_negative_values():
    """Test mean with negative values"""
    data = torch.tensor([[-1.0, -2.0, -3.0],
                         [-4.0, -5.0, -6.0]])
    result = ColwiseMean().solve(data)
    expected = torch.tensor([-2.5, -3.5, -4.5])
    assert torch.allclose(result, expected)


def test_mean_mixed_values():
    """Test mean with mixed positive and negative values"""
    data = torch.tensor([[-10.0, 0.0, 10.0],
                         [10.0, 0.0, -10.0]])
    result = ColwiseMean().solve(data)
    expected = torch.tensor([0.0, 0.0, 0.0])
    assert torch.allclose(result, expected)


def test_mean_empty_tensor():
    """Test mean of empty tensor returns empty tensor"""
    data = torch.tensor([])
    result = ColwiseMean().solve(data)
    assert result.numel() == 0


def test_mean_single_column():
    """Test mean with single column"""
    data = torch.tensor([[1.0],
                         [2.0],
                         [3.0],
                         [4.0]])
    result = ColwiseMean().solve(data)
    assert result.shape == (1,)
    assert abs(result[0].item() - 2.5) < 1e-6


def test_mean_dtype_preserved():
    """Test that dtype is preserved"""
    data = torch.tensor([[1.0, 2.0],
                         [3.0, 4.0]], dtype=torch.float64)
    result = ColwiseMean().solve(data)
    assert result.dtype == torch.float64
    expected = torch.tensor([2.0, 3.0], dtype=torch.float64)
    assert torch.allclose(result, expected)