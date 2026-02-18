# Custom autograd.Function: defining your own forward and backward passes
import torch
from torch.autograd import Function


# TODO: Implement a custom ReLU using autograd.Function
# - forward: return max(0, x) and save x for backward
# - backward: gradient is 1 where x > 0, else 0
class CustomReLU(Function):
    @staticmethod
    def forward(ctx, x):
        # TODO: Save x for backward, return relu output
        return None

    @staticmethod
    def backward(ctx, grad_output):
        # TODO: Return gradient only where x > 0
        return None


# TODO: Implement a custom function that clamps gradients during backward
# Forward: identity (just return x)
# Backward: clamp gradients to [-clip_value, clip_value]
class GradientClip(Function):
    @staticmethod
    def forward(ctx, x, clip_value):
        # TODO: Save clip_value for backward, return x unchanged
        return None

    @staticmethod
    def backward(ctx, grad_output):
        # TODO: Clamp the gradient and return it
        # Return None for the clip_value gradient (it's not a tensor)
        return None, None


# TODO: Implement Straight-Through Estimator (STE)
# Forward: binarize x (return 1 where x >= 0, else -1)
# Backward: pass gradients straight through (identity)
# This is used in quantization-aware training
class StraightThroughEstimator(Function):
    @staticmethod
    def forward(ctx, x):
        return None

    @staticmethod
    def backward(ctx, grad_output):
        return None


def use_custom_relu():
    x = torch.tensor([-1.0, 0.0, 1.0, 2.0], requires_grad=True)
    # TODO: Apply CustomReLU.apply to x
    output = None
    output.sum().backward()
    return output, x.grad


def use_gradient_clip():
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    output = GradientClip.apply(x, 0.5)
    # Create a large gradient
    loss = (output * 10).sum()
    loss.backward()
    # TODO: Return x.grad (should be clipped to [-0.5, 0.5])
    return x.grad


def use_ste():
    x = torch.tensor([-0.5, 0.0, 0.3, 1.0], requires_grad=True)
    # TODO: Apply StraightThroughEstimator.apply to x
    binary = None
    binary.sum().backward()
    return binary, x.grad


"""
----------------------TESTS-------------------------
------------------DO NOT TOUCH TESTS----------------
"""


def test_custom_relu():
    output, grad = use_custom_relu()
    assert torch.allclose(output, torch.tensor([0.0, 0.0, 1.0, 2.0]))
    assert torch.allclose(grad, torch.tensor([0.0, 0.0, 1.0, 1.0]))


def test_gradient_clip():
    grad = use_gradient_clip()
    assert torch.allclose(grad, torch.tensor([0.5, 0.5, 0.5]))


def test_ste():
    binary, grad = use_ste()
    assert torch.allclose(binary, torch.tensor([-1.0, 1.0, 1.0, 1.0]))
    # Straight-through: gradient is 1 everywhere
    assert torch.allclose(grad, torch.tensor([1.0, 1.0, 1.0, 1.0]))
