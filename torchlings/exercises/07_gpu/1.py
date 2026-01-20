# Device management: moving tensors between CPU and GPU
import torch
import torch.nn as nn

# These exercises require a CUDA GPU to run.
# If you don't have a GPU, you can run these on Modal.
import pytest

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


def check_cuda_available():
    # TODO: Return True if CUDA is available, False otherwise
    available = None
    return available


def get_device():
    # TODO: Return a torch.device for "cuda" if available, otherwise "cpu"
    device = None
    return device


def create_tensor_on_gpu():
    # TODO: Create a tensor directly on the GPU
    tensor = None
    return tensor


def move_tensor_to_gpu():
    cpu_tensor = torch.tensor([1.0, 2.0, 3.0])
    # TODO: Move the tensor to GPU using .to()
    gpu_tensor = None
    return gpu_tensor


def move_tensor_to_cpu():
    gpu_tensor = torch.tensor([1.0, 2.0, 3.0], device="cuda")
    # TODO: Move the tensor back to CPU
    cpu_tensor = None
    return cpu_tensor


def tensor_device_check():
    t = torch.tensor([1.0], device="cuda")
    # TODO: Return the device type as a string (e.g., "cuda")
    device_type = None
    return device_type


def operations_same_device():
    # Tensors must be on the same device for operations
    a = torch.tensor([1.0, 2.0], device="cuda")
    b = torch.tensor([3.0, 4.0])  # on CPU
    # TODO: Move b to the same device as a, then add them
    result = None
    return result


def gpu_memory_info():
    # TODO: Return the current GPU memory allocated in bytes
    # Hint: torch.cuda.memory_allocated()
    allocated = None
    return allocated


"""
----------------------TESTS-------------------------
------------------DO NOT TOUCH TESTS----------------
"""


def test_check_cuda_available():
    result = check_cuda_available()
    assert result == torch.cuda.is_available()


@requires_cuda
def test_get_device():
    device = get_device()
    assert device.type == "cuda"


@requires_cuda
def test_create_tensor_on_gpu():
    t = create_tensor_on_gpu()
    assert t.device.type == "cuda"


@requires_cuda
def test_move_tensor_to_gpu():
    t = move_tensor_to_gpu()
    assert t.device.type == "cuda"
    assert torch.allclose(t.cpu(), torch.tensor([1.0, 2.0, 3.0]))


@requires_cuda
def test_move_tensor_to_cpu():
    t = move_tensor_to_cpu()
    assert t.device.type == "cpu"


@requires_cuda
def test_tensor_device_check():
    dtype = tensor_device_check()
    assert dtype == "cuda"


@requires_cuda
def test_operations_same_device():
    result = operations_same_device()
    assert result.device.type == "cuda"
    assert torch.allclose(result.cpu(), torch.tensor([4.0, 6.0]))


@requires_cuda
def test_gpu_memory_info():
    allocated = gpu_memory_info()
    assert isinstance(allocated, int)
