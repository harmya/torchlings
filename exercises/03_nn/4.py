# Custom nn.Module: writing your own layers and models
import torch
import torch.nn as nn


# TODO: Define a custom module called TwoLayerNet
# It should have:
#   - __init__(self, input_dim, hidden_dim, output_dim)
#   - Two linear layers: fc1 (input -> hidden), fc2 (hidden -> output)
#   - A ReLU activation between them
#   - forward(self, x) that passes through fc1 -> relu -> fc2
class TwoLayerNet(nn.Module):
    pass


# TODO: Define a custom module called ResidualBlock
# It should have:
#   - __init__(self, dim)
#   - Two linear layers, both dim -> dim
#   - A ReLU activation
#   - forward(self, x) that computes: out = x + relu(fc2(relu(fc1(x))))
#   (this is a residual/skip connection)
class ResidualBlock(nn.Module):
    pass


def get_named_parameters():
    model = TwoLayerNet(4, 8, 2)
    # TODO: Return a list of (name, shape) tuples for all parameters
    # Hint: use model.named_parameters()
    param_info = None
    return param_info


def freeze_layer():
    model = TwoLayerNet(4, 8, 2)
    # TODO: Freeze fc1 so its parameters don't get gradients
    # Hint: set requires_grad = False on fc1's parameters

    return model


"""
----------------------TESTS-------------------------
------------------DO NOT TOUCH TESTS----------------
"""


def test_two_layer_net():
    model = TwoLayerNet(10, 20, 5)
    x = torch.randn(3, 10)
    out = model(x)
    assert out.shape == torch.Size([3, 5])
    assert hasattr(model, "fc1")
    assert hasattr(model, "fc2")


def test_residual_block():
    block = ResidualBlock(8)
    x = torch.randn(2, 8)
    out = block(x)
    assert out.shape == torch.Size([2, 8])
    # Output should differ from input (the block adds something)
    assert not torch.allclose(out, x)


def test_get_named_parameters():
    info = get_named_parameters()
    names = [name for name, _ in info]
    assert "fc1.weight" in names
    assert "fc1.bias" in names
    assert "fc2.weight" in names
    assert "fc2.bias" in names


def test_freeze_layer():
    model = freeze_layer()
    for param in model.fc1.parameters():
        assert param.requires_grad is False
    for param in model.fc2.parameters():
        assert param.requires_grad is True
