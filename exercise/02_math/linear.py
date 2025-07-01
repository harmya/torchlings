import torch
from exercise import Exercise

class LinearSystem2x2(Exercise):
    """
    Solve a 2×2 linear system Ax = b using Cramer's rule.
    For system: a11*x + a12*y = b1, a21*x + a22*y = b2
    Solution: x = det([[b1,a12],[b2,a22]]) / det(A)
              y = det([[a11,b1],[a21,b2]]) / det(A)
    
    Example:
        A = torch.tensor([[2.0, 1.0],
                         [1.0, 2.0]])
        b = torch.tensor([5.0, 4.0])
        result = solve(A, b)  # Returns: torch.tensor([2.0, 1.0])
        # Check: 2*2 + 1*1 = 5 ✓, 1*2 + 2*1 = 4 ✓
    """
    def solve(self, A, b):
        pass

"""TESTS: DO NOT MODIFY"""
def test_linear_basic():
    """Test basic linear system"""
    A = torch.tensor([[2.0, 1.0],
                      [1.0, 2.0]])
    b = torch.tensor([5.0, 4.0])
    result = LinearSystem2x2().solve(A, b)
    assert torch.allclose(result, torch.tensor([2.0, 1.0]))


def test_linear_identity():
    """Test with identity matrix"""
    A = torch.eye(2)
    b = torch.tensor([3.0, -2.0])
    result = LinearSystem2x2().solve(A, b)
    assert torch.allclose(result, b)


def test_linear_diagonal():
    """Test with diagonal matrix"""
    A = torch.tensor([[2.0, 0.0],
                      [0.0, 3.0]])
    b = torch.tensor([6.0, 9.0])
    result = LinearSystem2x2().solve(A, b)
    assert torch.allclose(result, torch.tensor([3.0, 3.0]))


def test_linear_negative():
    """Test with negative values"""
    A = torch.tensor([[1.0, -1.0],
                      [1.0, 1.0]])
    b = torch.tensor([2.0, 4.0])
    result = LinearSystem2x2().solve(A, b)
    # x + (-y) = 2, x + y = 4 => x = 3, y = 1
    assert torch.allclose(result, torch.tensor([3.0, 1.0]))


def test_linear_singular():
    """Test singular system returns NaN"""
    A = torch.tensor([[1.0, 2.0],
                      [2.0, 4.0]])  # Singular matrix
    b = torch.tensor([3.0, 6.0])
    result = LinearSystem2x2().solve(A, b)
    assert torch.all(torch.isnan(result))


def test_linear_verify_solution():
    """Test that solution satisfies Ax = b"""
    A = torch.tensor([[3.0, 2.0],
                      [1.0, 4.0]])
    b = torch.tensor([7.0, 5.0])
    x = LinearSystem2x2().solve(A, b)
    # Verify Ax = b
    result = torch.mv(A, x)
    assert torch.allclose(result, b, atol=1e-6)


def test_linear_large_values():
    """Test with large coefficient values"""
    A = torch.tensor([[100.0, 50.0],
                      [25.0, 75.0]])
    b = torch.tensor([300.0, 200.0])
    x = LinearSystem2x2().solve(A, b)
    # Verify solution
    result = torch.mv(A, x)
    assert torch.allclose(result, b, atol=1e-4)