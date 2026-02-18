# Triton kernels: writing custom GPU kernels
import torch
import pytest

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

requires_triton = pytest.mark.skipif(not HAS_TRITON, reason="Triton not available")


if HAS_TRITON:

    # TODO: Write a Triton kernel that adds two vectors element-wise
    # This is the "hello world" of Triton
    @triton.jit
    def vector_add_kernel(
        x_ptr,
        y_ptr,
        output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        # TODO: Implement the kernel
        # 1. Get the program ID (block index): tl.program_id(0)
        # 2. Compute element offsets for this block
        # 3. Create a mask for bounds checking
        # 4. Load x and y values
        # 5. Compute x + y
        # 6. Store the result
        pass

    # TODO: Write a Triton kernel that applies ReLU element-wise
    @triton.jit
    def relu_kernel(
        x_ptr,
        output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        # TODO: Implement ReLU: max(0, x)
        # Use tl.maximum(x, 0) or tl.where(x > 0, x, 0)
        pass

    # TODO: Write a Triton kernel for fused multiply-add: a * x + b
    @triton.jit
    def fused_mul_add_kernel(
        x_ptr,
        a_ptr,
        b_ptr,
        output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        # TODO: Compute a * x + b in one kernel (avoids intermediate memory)
        pass


def triton_vector_add(x, y):
    assert x.is_cuda and y.is_cuda
    output = torch.empty_like(x)
    n = x.numel()
    # TODO: Launch vector_add_kernel with the right grid
    # grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    grid = None
    vector_add_kernel[grid](x, y, output, n, BLOCK_SIZE=1024)
    return output


def triton_relu(x):
    assert x.is_cuda
    output = torch.empty_like(x)
    n = x.numel()
    # TODO: Launch relu_kernel
    grid = None
    relu_kernel[grid](x, output, n, BLOCK_SIZE=1024)
    return output


def triton_fused_mul_add(x, a, b):
    assert x.is_cuda
    output = torch.empty_like(x)
    n = x.numel()
    # TODO: Launch fused_mul_add_kernel
    grid = None
    fused_mul_add_kernel[grid](x, a, b, output, n, BLOCK_SIZE=1024)
    return output


"""
----------------------TESTS-------------------------
------------------DO NOT TOUCH TESTS----------------
"""


@requires_cuda
@requires_triton
def test_triton_vector_add():
    x = torch.randn(1024, device="cuda")
    y = torch.randn(1024, device="cuda")
    result = triton_vector_add(x, y)
    expected = x + y
    assert torch.allclose(result, expected, atol=1e-5)


@requires_cuda
@requires_triton
def test_triton_relu():
    x = torch.randn(1024, device="cuda")
    result = triton_relu(x)
    expected = torch.relu(x)
    assert torch.allclose(result, expected, atol=1e-5)


@requires_cuda
@requires_triton
def test_triton_fused_mul_add():
    x = torch.randn(1024, device="cuda")
    a = torch.tensor(2.5, device="cuda")
    b = torch.tensor(-1.0, device="cuda")
    result = triton_fused_mul_add(x, a, b)
    expected = 2.5 * x + (-1.0)
    assert torch.allclose(result, expected, atol=1e-4)
