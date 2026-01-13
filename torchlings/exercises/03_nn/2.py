# Activation functions: adding non-linearity to networks
import torch
import torch.nn as nn
import torch.nn.functional as F


def apply_relu():
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    # TODO: Apply ReLU activation to x
    result = None
    return result


def apply_sigmoid():
    x = torch.tensor([-2.0, 0.0, 2.0])
    # TODO: Apply sigmoid activation to x
    result = None
    return result


def apply_tanh():
    x = torch.tensor([-1.0, 0.0, 1.0])
    # TODO: Apply tanh activation to x
    result = None
    return result


def apply_softmax():
    logits = torch.tensor([2.0, 1.0, 0.1])
    # TODO: Apply softmax along dim=0 so outputs sum to 1
    probs = None
    return probs


def apply_gelu():
    x = torch.tensor([-1.0, 0.0, 1.0])
    # TODO: Apply GELU activation to x using the functional API (F.gelu)
    result = None
    return result


def module_vs_functional():
    x = torch.tensor([-1.0, 0.0, 1.0, 2.0])
    # There are two ways to apply activations in PyTorch:
    # 1. nn.Module (stateful, used in model definitions)
    # 2. torch.nn.functional (stateless, used in forward methods)

    # TODO: Apply ReLU using the module API (nn.ReLU)
    relu_module = None
    result_module = None

    # TODO: Apply ReLU using the functional API (F.relu)
    result_functional = None

    return result_module, result_functional


def leaky_relu():
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    # TODO: Apply LeakyReLU with negative_slope=0.1
    result = None
    return result


"""
----------------------TESTS-------------------------
------------------DO NOT TOUCH TESTS----------------
"""


def test_apply_relu():
    result = apply_relu()
    expected = torch.tensor([0.0, 0.0, 0.0, 1.0, 2.0])
    assert torch.allclose(result, expected)


def test_apply_sigmoid():
    result = apply_sigmoid()
    assert result.shape == torch.Size([3])
    assert torch.all(result >= 0) and torch.all(result <= 1)
    assert torch.allclose(result[1], torch.tensor(0.5))


def test_apply_tanh():
    result = apply_tanh()
    assert result.shape == torch.Size([3])
    assert torch.all(result >= -1) and torch.all(result <= 1)
    assert torch.allclose(result[1], torch.tensor(0.0))


def test_apply_softmax():
    probs = apply_softmax()
    assert torch.allclose(probs.sum(), torch.tensor(1.0))
    assert torch.all(probs > 0)


def test_apply_gelu():
    result = apply_gelu()
    assert result.shape == torch.Size([3])
    assert torch.allclose(result[1], torch.tensor(0.0))
    assert result[2] > result[0]


def test_module_vs_functional():
    r_mod, r_func = module_vs_functional()
    expected = torch.tensor([0.0, 0.0, 1.0, 2.0])
    assert torch.allclose(r_mod, expected)
    assert torch.allclose(r_func, expected)


def test_leaky_relu():
    result = leaky_relu()
    expected = torch.tensor([-0.2, -0.1, 0.0, 1.0, 2.0])
    assert torch.allclose(result, expected)
