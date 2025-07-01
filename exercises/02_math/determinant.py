import torch
from exercise import Exercise

class Determinant2x2(Exercise):
    """
    Compute determinant of a 2×2 matrix: det([[a,b],[c,d]]) = ad - bc
    
    Example:
        mat = torch.tensor([[3.0, 8.0],
                           [4.0, 6.0]])
        result = solve(mat)  # Returns: 3*6 - 8*4 = 18 - 32 = -14
    """
    def solve(self, mat):
        pass

"""TESTS: DO NOT MODIFY"""

def test_det_basic():
    """Test basic determinant calculation"""
    mat = torch.tensor([[3.0, 8.0],
                        [4.0, 6.0]])
    result = Determinant2x2().solve(mat)
    assert abs(result.item() - (-14.0)) < 1e-6


def test_det_identity():
    """Test determinant of identity matrix"""
    mat = torch.eye(2)
    result = Determinant2x2().solve(mat)
    assert abs(result.item() - 1.0) < 1e-6


def test_det_zero():
    """Test determinant of singular matrix"""
    mat = torch.tensor([[1.0, 2.0],
                        [2.0, 4.0]])  # Second row is 2x first row
    result = Determinant2x2().solve(mat)
    assert abs(result.item()) < 1e-6


def test_det_negative():
    """Test determinant with negative values"""
    mat = torch.tensor([[-1.0, 2.0],
                        [3.0, -4.0]])
    result = Determinant2x2().solve(mat)
    # det = (-1)*(-4) - 2*3 = 4 - 6 = -2
    assert abs(result.item() - (-2.0)) < 1e-6


def test_det_integer():
    """Test determinant with integer matrix"""
    mat = torch.tensor([[5, 2],
                        [3, 1]], dtype=torch.int32)
    result = Determinant2x2().solve(mat)
    assert result.item() == -1


def test_det_scale():
    """Test that scaling a row scales the determinant"""
    mat1 = torch.tensor([[1.0, 2.0],
                         [3.0, 4.0]])
    mat2 = torch.tensor([[2.0, 4.0],  # First row scaled by 2
                         [3.0, 4.0]])
    det1 = Determinant2x2().solve(mat1)
    det2 = Determinant2x2().solve(mat2)
    assert abs(det2.item() - 2 * det1.item()) < 1e-6


def test_det_transpose():
    """Test that transpose preserves determinant"""
    mat = torch.tensor([[3.0, 5.0],
                        [1.0, 2.0]])
    det_original = Determinant2x2().solve(mat)
    det_transpose = Determinant2x2().solve(mat.T)
    assert abs(det_original.item() - det_transpose.item()) < 1e-6