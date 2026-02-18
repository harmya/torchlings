# Linear layers: the building block of neural networks
import torch
import torch.nn as nn


def create_linear_layer():
    # TODO: Create a linear layer that maps from 10 input features to 5 output features
    layer = None
    return layer


def linear_layer_shapes():
    layer = nn.Linear(8, 4)
    # TODO: Return the shape of the weight matrix and the shape of the bias vector
    weight_shape = None
    bias_shape = None
    return weight_shape, bias_shape


def linear_no_bias():
    # TODO: Create a linear layer from 6 inputs to 3 outputs with no bias term
    layer = None
    return layer


def linear_forward_pass():
    torch.manual_seed(42)
    layer = nn.Linear(4, 2)
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    # TODO: Pass x through the layer and return the output
    output = None
    return output


def count_parameters():
    layer = nn.Linear(10, 5)
    # TODO: Count the total number of trainable parameters (weight + bias)
    total = None
    return total


def manual_linear():
    # A linear layer computes y = xW^T + b
    # TODO: Manually compute what nn.Linear does using matrix operations
    torch.manual_seed(0)
    layer = nn.Linear(3, 2)
    x = torch.tensor([[1.0, 2.0, 3.0]])

    # Use layer.weight and layer.bias to compute the output manually
    output = None
    return output, layer(x)


"""
----------------------TESTS-------------------------
------------------DO NOT TOUCH TESTS----------------
"""


def test_create_linear_layer():
    layer = create_linear_layer()
    assert isinstance(layer, nn.Linear)
    assert layer.in_features == 10
    assert layer.out_features == 5


def test_linear_layer_shapes():
    w_shape, b_shape = linear_layer_shapes()
    assert w_shape == torch.Size([4, 8])
    assert b_shape == torch.Size([4])


def test_linear_no_bias():
    layer = linear_no_bias()
    assert isinstance(layer, nn.Linear)
    assert layer.in_features == 6
    assert layer.out_features == 3
    assert layer.bias is None


def test_linear_forward_pass():
    output = linear_forward_pass()
    assert output.shape == torch.Size([1, 2])


def test_count_parameters():
    total = count_parameters()
    assert total == 55  # 10*5 + 5


def test_manual_linear():
    manual, builtin = manual_linear()
    assert torch.allclose(manual, builtin, atol=1e-5)
