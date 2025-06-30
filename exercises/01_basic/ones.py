import torch
import pytest

def ones(shape):
    return torch.ones(shape)



"""TESTS: DO NOT MODIFY"""
def test_ones():
    """Test that the ones function returns a tensor of ones"""
    assert torch.all(ones(5) == torch.ones(5))


def test_ones_shape():
    """Test that the ones function returns a tensor of the correct shape"""
    assert ones(5).shape == (5,)


def test_ones_type():
    """Test that the ones function returns a tensor of the correct type"""
    assert ones(5).dtype == torch.float32


def test_ones_values():
    """Test that all values in the tensor are actually 1.0"""
    result = ones(10)
    assert torch.all(result == 1.0)
    assert result.sum().item() == 10.0


def test_ones_zero_size():
    """Test creating a tensor with size 0"""
    result = ones(0)
    assert result.shape == (0,)
    assert result.numel() == 0


def test_ones_large_size():
    """Test creating a large tensor"""
    result = ones(1000)
    assert result.shape == (1000,)
    assert torch.all(result == 1.0)


def test_ones_negative_size():
    """Test that negative size raises an error"""
    with pytest.raises(RuntimeError):
        ones(-5)
