# Model device management: moving models to GPU
import torch
import torch.nn as nn
import pytest

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


def move_model_to_gpu():
    model = nn.Linear(10, 5)
    # TODO: Move the entire model to GPU
    # Unlike tensors, model.to() modifies the model in-place
    return model


def model_parameters_on_device():
    model = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 2),
    )
    model.to("cuda")
    # TODO: Check that ALL parameters are on GPU
    # Return True if all parameters are on cuda, False otherwise
    all_on_gpu = None
    return all_on_gpu


def forward_pass_on_gpu():
    torch.manual_seed(42)
    model = nn.Linear(4, 2).to("cuda")
    # TODO: Create input tensor ON GPU and run a forward pass
    x = None
    output = None
    return output


def multi_gpu_count():
    # TODO: Return the number of available CUDA devices
    count = None
    return count


def pin_memory_transfer():
    # Pinned memory enables faster CPU->GPU transfers
    # TODO: Create a pinned memory tensor and move it to GPU
    cpu_tensor = None  # Hint: torch.randn(...).pin_memory()
    gpu_tensor = None  # Move to GPU with non_blocking=True
    return gpu_tensor


def device_agnostic_model():
    # TODO: Write device-agnostic code that works on both CPU and GPU
    # Create a device based on availability
    device = None
    model = nn.Linear(4, 2).to(device)
    x = torch.randn(3, 4).to(device)
    output = model(x)
    return output, device


"""
----------------------TESTS-------------------------
------------------DO NOT TOUCH TESTS----------------
"""


@requires_cuda
def test_move_model_to_gpu():
    model = move_model_to_gpu()
    assert next(model.parameters()).device.type == "cuda"


@requires_cuda
def test_model_parameters_on_device():
    result = model_parameters_on_device()
    assert result is True


@requires_cuda
def test_forward_pass_on_gpu():
    output = forward_pass_on_gpu()
    assert output.device.type == "cuda"
    assert output.shape[-1] == 2


@requires_cuda
def test_multi_gpu_count():
    count = multi_gpu_count()
    assert count >= 1


@requires_cuda
def test_pin_memory_transfer():
    t = pin_memory_transfer()
    assert t.device.type == "cuda"


def test_device_agnostic_model():
    output, device = device_agnostic_model()
    assert output.device == device
    assert output.shape == torch.Size([3, 2])
