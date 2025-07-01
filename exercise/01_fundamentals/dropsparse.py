import torch

from exercise import Exercise

class DropSparsestColumn(Exercise):
    """
    Remove the column with the most zeros from a 2D matrix.
    If multiple columns have the same number of zeros, remove the first one.
    
    Example:
        mat = torch.tensor([[0, 1, 0],
                           [0, 2, 3]])
        # Column 0 has 2 zeros, column 1 has 0 zeros, column 2 has 1 zero
        # Drop column 0 (most zeros)
        result = solve(mat)  # Returns: torch.tensor([[1, 0], [2, 3]])
    """
    def solve(self, mat):
        pass

"""TESTS: DO NOT MODIFY"""
def test_drop_basic():
    """Test dropping the sparsest column"""
    mat = torch.tensor([[0,1,0],[0,2,3]])
    # zero_counts = [2,0,1] ⇒ drop col 0
    out = DropSparsestColumn().solve(mat)
    assert out.shape == (2,2)
    assert torch.equal(out, torch.tensor([[1,0],[2,3]]))

def test_drop_tie_first():
    """Test that ties are broken by choosing the first column"""
    mat = torch.tensor([[0,1],[2,0]])
    # zero_counts = [1,1] ⇒ drop first (col 0)
    out = DropSparsestColumn().solve(mat)
    assert out.shape == (2,1)
    assert torch.equal(out, torch.tensor([[1],[0]]))

def test_drop_single_column():
    """Test dropping the only column"""
    mat = torch.ones(3,1)
    out = DropSparsestColumn().solve(mat)
    assert out.numel() == 0 and out.shape == (3,0)

def test_drop_no_zero_column():
    """Test when no column has zeros"""
    mat = torch.arange(6).view(2,3) + 1  # all nonzero
    out = DropSparsestColumn().solve(mat)
    # zero_counts all 0 ⇒ drop col 0
    assert out.shape == (2,2)
    assert torch.equal(out, mat[:,1:])

def test_drop_dtype_preserved():
    """Test that dtype is preserved"""
    mat = torch.tensor([[0,0],[0,1]], dtype=torch.int32)
    out = DropSparsestColumn().solve(mat)
    assert out.dtype == torch.int32

def test_drop_all_zeros():
    """Test matrix with all zeros"""
    mat = torch.zeros(3, 3)
    out = DropSparsestColumn().solve(mat)
    # All columns have 3 zeros, drop column 0
    assert out.shape == (3, 2)
    assert torch.equal(out, mat[:, 1:])

def test_drop_float_zeros():
    """Test with floating point zeros"""
    mat = torch.tensor([[0.0, 1.0, 0.0],
                        [2.0, 0.0, 0.0],
                        [0.0, 3.0, 0.0]])
    # Column 0 has 2 zeros, column 1 has 1 zero, column 2 has 3 zeros
    # Drop column 2
    out = DropSparsestColumn().solve(mat)
    assert out.shape == (3, 2)
    assert torch.equal(out, mat[:, :2])