import torch
from exercise import Exercise

class AlternateRowSum(Exercise):
    """
    Compute the sum of each alternate row (rows 0, 2, 4, ...) in a 2D matrix.
    
    Example:
        mat = torch.tensor([[1, 2],    # row 0: sum = 3
                           [3, 4],     # row 1: skip
                           [5, 6],     # row 2: sum = 11
                           [7, 8]])    # row 3: skip
        result = solve(mat)  # Returns: torch.tensor([3, 11])
    """
    def solve(self, mat):
        pass

"""TESTS: DO NOT MODIFY"""
def test_alt_sum_basic():
    """Test basic alternate row summation"""
    mat = torch.tensor([[1,2],[3,4],[5,6]])
    out = AlternateRowSum().solve(mat)
    # rows 0 and 2: sums 3 and 11
    assert torch.equal(out, torch.tensor([3,11]))

def test_alt_sum_single_row():
    """Test with single row matrix"""
    mat = torch.tensor([[7,8]])
    out = AlternateRowSum().solve(mat)
    assert out.tolist() == [15]

def test_alt_sum_empty():
    """Test with empty matrix"""
    mat = torch.zeros(0,5)
    out = AlternateRowSum().solve(mat)
    assert out.numel() == 0 and out.shape == (0,)

def test_alt_sum_dtype():
    """Test that dtype is preserved"""
    mat = torch.ones(2,3, dtype=torch.int64)
    out = AlternateRowSum().solve(mat)
    assert out.dtype == torch.int64

def test_alt_sum_random():
    """Test with random values"""
    mat = torch.randn(6,4)
    out = AlternateRowSum().solve(mat)
    # out[i] == sum of mat[2*i]
    assert torch.allclose(out, mat[::2].sum(dim=1))

def test_alt_sum_odd_rows():
    """Test with odd number of rows"""
    mat = torch.arange(15).reshape(5, 3)
    out = AlternateRowSum().solve(mat)
    # Rows 0, 2, 4: sums are 3, 21, 39
    assert torch.equal(out, torch.tensor([3, 21, 39]))

def test_alt_sum_large_matrix():
    """Test with large matrix"""
    mat = torch.ones(100, 50)
    out = AlternateRowSum().solve(mat)
    assert out.shape == (50,)  # 50 alternate rows
    assert torch.all(out == 50.0)  # Each row sums to 50