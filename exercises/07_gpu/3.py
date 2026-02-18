# Mixed precision: faster training with lower precision
import torch
import torch.nn as nn
import pytest

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


def float16_tensor():
    # TODO: Create a float16 (half precision) tensor on GPU
    tensor = None
    return tensor


def bfloat16_tensor():
    # TODO: Create a bfloat16 tensor on GPU
    # bfloat16 has the same range as float32 but less precision
    tensor = None
    return tensor


def autocast_forward():
    # torch.autocast automatically chooses precision for each operation
    model = nn.Linear(256, 128).cuda()
    x = torch.randn(32, 256, device="cuda")
    # TODO: Run the forward pass inside a torch.autocast context for "cuda"
    # Return the output and its dtype
    output = None
    return output


def autocast_preserves_master_weights():
    model = nn.Linear(64, 32).cuda()
    # TODO: Verify that model weights stay in float32 even after autocast forward
    x = torch.randn(8, 64, device="cuda")
    with torch.autocast("cuda"):
        output = model(x)
    # Return the dtype of the model weights and the output dtype
    weight_dtype = None
    output_dtype = None
    return weight_dtype, output_dtype


def grad_scaler_training():
    # GradScaler prevents underflow in float16 gradients
    model = nn.Linear(128, 64).cuda()
    optimizer = torch.optim.Adam(model.parameters())
    scaler = torch.amp.GradScaler("cuda")
    x = torch.randn(16, 128, device="cuda")
    target = torch.randn(16, 64, device="cuda")

    # TODO: Implement one training step with mixed precision:
    # 1. Zero gradients
    # 2. Forward pass inside autocast
    # 3. Scale the loss with scaler.scale(loss)
    # 4. Call backward on the scaled loss
    # 5. scaler.step(optimizer)
    # 6. scaler.update()
    loss = None

    return loss


def cast_between_dtypes():
    # TODO: Create a float32 tensor, cast to float16, then back to float32
    original = torch.randn(4, 4, device="cuda")
    half = None
    back_to_float = None
    return original.dtype, half.dtype, back_to_float.dtype


"""
----------------------TESTS-------------------------
------------------DO NOT TOUCH TESTS----------------
"""


@requires_cuda
def test_float16_tensor():
    t = float16_tensor()
    assert t.dtype == torch.float16
    assert t.device.type == "cuda"


@requires_cuda
def test_bfloat16_tensor():
    t = bfloat16_tensor()
    assert t.dtype == torch.bfloat16
    assert t.device.type == "cuda"


@requires_cuda
def test_autocast_forward():
    output = autocast_forward()
    assert output.device.type == "cuda"
    # Autocast typically uses float16 or bfloat16 for linear layers
    assert output.dtype in (torch.float16, torch.bfloat16)


@requires_cuda
def test_autocast_preserves_master_weights():
    w_dtype, o_dtype = autocast_preserves_master_weights()
    assert w_dtype == torch.float32
    assert o_dtype in (torch.float16, torch.bfloat16)


@requires_cuda
def test_grad_scaler_training():
    loss = grad_scaler_training()
    assert loss is not None
    assert loss.item() > 0


@requires_cuda
def test_cast_between_dtypes():
    orig, half, back = cast_between_dtypes()
    assert orig == torch.float32
    assert half == torch.float16
    assert back == torch.float32
