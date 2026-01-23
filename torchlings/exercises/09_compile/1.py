# torch.compile basics: JIT compilation for faster models
import torch
import torch.nn as nn
import pytest

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


def compile_simple_function():
    # torch.compile traces and optimizes a function or module
    def my_fn(x):
        return x * x + 2 * x + 1

    # TODO: Compile my_fn using torch.compile
    compiled_fn = None
    return compiled_fn


def compile_model():
    model = nn.Sequential(
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )
    # TODO: Compile the model with torch.compile
    compiled_model = None
    return compiled_model


def compile_with_mode():
    model = nn.Linear(64, 32)
    # TODO: Compile with mode="reduce-overhead" for small models
    # Other modes: "default", "max-autotune"
    compiled = None
    return compiled


def compiled_produces_same_output():
    torch.manual_seed(42)
    model = nn.Linear(16, 8)
    x = torch.randn(4, 16)

    eager_out = model(x)
    # TODO: Compile the model and run the same input
    compiled_model = None
    compiled_out = None

    return eager_out, compiled_out


def compile_with_fullgraph():
    # fullgraph=True requires the entire function to be captured in one graph
    # (no graph breaks allowed)
    def simple_fn(x, y):
        return torch.matmul(x, y.T) + 1.0

    # TODO: Compile with fullgraph=True
    compiled = None
    return compiled


"""
----------------------TESTS-------------------------
------------------DO NOT TOUCH TESTS----------------
"""


def test_compile_simple_function():
    fn = compile_simple_function()
    x = torch.tensor([1.0, 2.0, 3.0])
    result = fn(x)
    expected = x * x + 2 * x + 1
    assert torch.allclose(result, expected)


def test_compile_model():
    model = compile_model()
    x = torch.randn(2, 64)
    output = model(x)
    assert output.shape == torch.Size([2, 10])


def test_compile_with_mode():
    model = compile_with_mode()
    x = torch.randn(4, 64)
    output = model(x)
    assert output.shape == torch.Size([4, 32])


def test_compiled_produces_same_output():
    eager, compiled = compiled_produces_same_output()
    assert torch.allclose(eager, compiled, atol=1e-5)


def test_compile_with_fullgraph():
    fn = compile_with_fullgraph()
    x = torch.randn(3, 4)
    y = torch.randn(5, 4)
    result = fn(x, y)
    expected = torch.matmul(x, y.T) + 1.0
    assert torch.allclose(result, expected, atol=1e-5)
