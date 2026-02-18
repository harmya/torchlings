# Compile modes and backends: tuning compilation for your workload
import torch
import torch.nn as nn
import pytest

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


class TransformerBlock(nn.Module):
    def __init__(self, dim=64, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.ffn(self.norm2(x))
        return x


def compile_default_mode():
    model = TransformerBlock()
    # TODO: Compile with mode="default" (balanced compile time vs performance)
    compiled = None
    return compiled


def compile_reduce_overhead():
    model = TransformerBlock()
    # TODO: Compile with mode="reduce-overhead"
    # This uses CUDA graphs to reduce kernel launch overhead
    # Best for small batch sizes with many kernel launches
    compiled = None
    return compiled


def compile_max_autotune():
    model = TransformerBlock()
    # TODO: Compile with mode="max-autotune"
    # This tries many implementations and picks the fastest
    # Slowest compile time, fastest runtime
    compiled = None
    return compiled


def torch_compile_reset():
    # torch._dynamo.reset() clears the compile cache
    # Useful when you want to recompile with different settings
    # TODO: Reset the dynamo cache
    torch._dynamo.reset()

    model = nn.Linear(32, 16)
    # TODO: Compile and run
    compiled = None
    x = torch.randn(4, 32)
    output = None
    return output


def disable_compile():
    # torch.compiler.disable lets you mark regions that should not be compiled
    # Useful for debugging or code that can't be traced

    @torch.compiler.disable
    def debug_print(x):
        print(f"Shape: {x.shape}, Mean: {x.mean().item():.4f}")
        return x

    def model_fn(x):
        x = x * 2
        x = debug_print(x)  # this part won't be compiled
        x = x + 1
        return x

    # TODO: Compile the model_fn and run it
    compiled = None
    x = torch.randn(4, 4)
    output = None
    return output


"""
----------------------TESTS-------------------------
------------------DO NOT TOUCH TESTS----------------
"""


def test_compile_default_mode():
    model = compile_default_mode()
    x = torch.randn(2, 8, 64)
    output = model(x)
    assert output.shape == torch.Size([2, 8, 64])


def test_compile_reduce_overhead():
    model = compile_reduce_overhead()
    x = torch.randn(2, 8, 64)
    output = model(x)
    assert output.shape == torch.Size([2, 8, 64])


def test_compile_max_autotune():
    model = compile_max_autotune()
    x = torch.randn(2, 8, 64)
    output = model(x)
    assert output.shape == torch.Size([2, 8, 64])


def test_torch_compile_reset():
    output = torch_compile_reset()
    assert output.shape == torch.Size([4, 16])


def test_disable_compile():
    output = disable_compile()
    expected = torch.randn(4, 4)  # won't match, just check shape
    assert output.shape == torch.Size([4, 4])
