import torch
from exercise import Exercise

class Eigenvalues2x2(Exercise):
    """
    Compute eigenvalues of a 2×2 matrix via characteristic polynomial.
    For matrix [[a,b],[c,d]], characteristic equation is: λ² - (a+d)λ + (ad-bc) = 0
    Using quadratic formula: λ = ((a+d) ± √((a+d)² - 4(ad-bc))) / 2
    
    Example:
        mat = torch.tensor([[3.0, 1.0],
                           [1.0, 3.0]])
        result = solve(mat)  # Returns: torch.tensor([4.0, 2.0])
    """
    def solve(self, mat):
        pass

"""TESTS: DO NOT MODIFY"""

def test_eigen_symmetric():
    """Test eigenvalues of symmetric matrix"""
    mat = torch.tensor([[3.0, 1.0],
                        [1.0, 3.0]])
    result = Eigenvalues2x2().solve(mat)
    # Eigenvalues should be 4 and 2
    eigenvals = sorted(result.tolist())
    assert abs(eigenvals[0] - 2.0) < 1e-6
    assert abs(eigenvals[1] - 4.0) < 1e-6


def test_eigen_identity():
    """Test eigenvalues of identity matrix"""
    mat = torch.eye(2)
    result = Eigenvalues2x2().solve(mat)
    # Both eigenvalues should be 1
    assert torch.allclose(result, torch.ones(2))


def test_eigen_diagonal():
    """Test eigenvalues of diagonal matrix"""
    mat = torch.tensor([[5.0, 0.0],
                        [0.0, 3.0]])
    result = Eigenvalues2x2().solve(mat)
    # Eigenvalues should be the diagonal elements
    eigenvals = sorted(result.tolist())
    assert abs(eigenvals[0] - 3.0) < 1e-6
    assert abs(eigenvals[1] - 5.0) < 1e-6


def test_eigen_zero_matrix():
    """Test eigenvalues of zero matrix"""
    mat = torch.zeros(2, 2)
    result = Eigenvalues2x2().solve(mat)
    # Both eigenvalues should be 0
    assert torch.allclose(result, torch.zeros(2))


def test_eigen_singular():
    """Test eigenvalues of singular matrix"""
    mat = torch.tensor([[1.0, 2.0],
                        [2.0, 4.0]])
    result = Eigenvalues2x2().solve(mat)
    # One eigenvalue should be 0, other should be 5
    eigenvals = sorted(result.tolist())
    assert abs(eigenvals[0]) < 1e-6
    assert abs(eigenvals[1] - 5.0) < 1e-6


def test_eigen_sum_product():
    """Test that sum of eigenvalues equals trace and product equals determinant"""
    mat = torch.tensor([[4.0, -2.0],
                        [1.0, 1.0]])
    result = Eigenvalues2x2().solve(mat)
    trace = mat[0, 0] + mat[1, 1]
    det = mat[0, 0] * mat[1, 1] - mat[0, 1] * mat[1, 0]
    assert abs(result.sum().item() - trace) < 1e-6
    assert abs(result.prod().item() - det) < 1e-6


def test_eigen_complex_case():
    """Test matrix with complex eigenvalues returns NaN"""
    mat = torch.tensor([[0.0, 1.0],
                        [-1.0, 0.0]])  # Rotation matrix
    result = Eigenvalues2x2().solve(mat)
    assert torch.all(torch.isnan(result))