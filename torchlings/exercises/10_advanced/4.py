# torch.export and memory management
import torch
import torch.nn as nn
import tempfile
import os
import pytest

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


def export_simple_model():
    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 2),
    )
    model.eval()
    # TODO: Export the model using torch.export.export
    # The batch dimension should be dynamic so the exported model
    # works with any batch size. Use torch.export.Dim to mark it.
    # Note: nn.Sequential's forward arg is called "input"
    # Hint: batch = torch.export.Dim("batch", min=1)
    #       dynamic_shapes = {"input": {0: batch}}
    # Use batch_size >= 2 in the example input (size 1 gets specialized)
    example_input = (torch.randn(2, 4),)
    exported = None
    return exported


def export_and_run():
    model = nn.Sequential(nn.Linear(4, 2))
    model.eval()
    batch = torch.export.Dim("batch", min=1)
    example_input = (torch.randn(2, 4),)
    dynamic_shapes = {"input": {0: batch}}
    # TODO: Export the model with dynamic shapes
    exported = None
    # TODO: Run the exported program with a new input
    x = torch.randn(3, 4)
    output = None
    return output


def trace_model():
    # torch.jit.trace captures a model by running example input through it
    model = nn.Sequential(nn.Linear(4, 2), nn.ReLU())
    model.eval()
    example_input = torch.randn(1, 4)
    # TODO: Trace the model using torch.jit.trace
    traced = None
    return traced


def save_load_traced():
    model = nn.Sequential(nn.Linear(4, 2))
    model.eval()
    traced = torch.jit.trace(model, torch.randn(1, 4))

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    try:
        # TODO: Save the traced model with torch.jit.save
        # TODO: Load it back with torch.jit.load
        loaded = None
        x = torch.randn(2, 4)
        return loaded(x)
    finally:
        os.unlink(path)


def memory_allocated():
    # TODO: Return current CUDA memory allocated in bytes
    allocated = None
    return allocated


def empty_cache():
    # When tensors are deleted, GPU memory isn't immediately freed
    # torch.cuda.empty_cache() releases it back to the OS
    x = torch.randn(1000, 1000, device="cuda")
    del x
    # TODO: Empty the CUDA cache
    # TODO: Return memory allocated after emptying cache
    allocated = None
    return allocated


"""
----------------------TESTS-------------------------
------------------DO NOT TOUCH TESTS----------------
"""


def test_export_simple_model():
    exported = export_simple_model()
    assert exported is not None
    x = torch.randn(2, 4)
    result = exported.module()(x)
    assert result.shape == torch.Size([2, 2])


def test_export_and_run():
    output = export_and_run()
    assert output.shape == torch.Size([3, 2])


def test_trace_model():
    traced = trace_model()
    x = torch.randn(5, 4)
    output = traced(x)
    assert output.shape == torch.Size([5, 2])


def test_save_load_traced():
    output = save_load_traced()
    assert output.shape == torch.Size([2, 2])


@requires_cuda
def test_memory_allocated():
    allocated = memory_allocated()
    assert isinstance(allocated, int)


@requires_cuda
def test_empty_cache():
    allocated = empty_cache()
    assert isinstance(allocated, int)
