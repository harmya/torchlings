import torch
import pytest
from exercise import Exercise

class MatVecMul(Exercise):
    def solve(self, mat, vec):
        pass

"""TESTS: DO NOT MODIFY"""
def test_matvec_basic():
    mat = torch.tensor([[1,2],[3,4]], dtype=torch.float32)
    vec = torch.tensor([1,1], dtype=torch.float32)
    out = MatVecMul().solve(mat, vec)
    assert torch.equal(out, torch.tensor([3,7]))

def test_matvec_shape():
    m, n = 5, 3
    mat = torch.zeros(m,n)
    vec = torch.zeros(n)
    out = MatVecMul().solve(mat, vec)
    assert out.shape == (m,)

def test_matvec_dtype_preserved():
    mat = torch.ones(2,2, dtype=torch.int64)
    vec = torch.ones(2, dtype=torch.int64)
    out = MatVecMul().solve(mat, vec)
    assert out.dtype == torch.int64

def test_matvec_mismatch_error():
    with pytest.raises(RuntimeError):
        MatVecMul().solve(torch.zeros(2,3), torch.zeros(2))

def test_matvec_zero_vector():
    mat = torch.randn(4,4)
    vec = torch.zeros(4)
    out = MatVecMul().solve(mat, vec)
    assert torch.all(out == 0)
