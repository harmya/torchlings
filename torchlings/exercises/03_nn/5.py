# Convolutions: spatial feature extraction
import torch
import torch.nn as nn


def create_conv2d():
    # TODO: Create a Conv2d layer:
    #   1 input channel, 16 output channels, kernel size 3, padding 1
    conv = None
    return conv


def conv_output_shape():
    # A Conv2d with no padding reduces spatial dimensions
    conv = nn.Conv2d(3, 16, kernel_size=5)  # no padding
    x = torch.randn(1, 3, 32, 32)  # batch=1, channels=3, height=32, width=32
    # TODO: Pass x through conv and return the output shape
    output_shape = None
    return output_shape


def conv_with_stride():
    # TODO: Create a Conv2d that halves spatial dimensions:
    #   3 input channels, 8 output channels, kernel size 3, stride 2, padding 1
    conv = None
    x = torch.randn(1, 3, 16, 16)
    # TODO: Pass x through conv and return the output shape
    output_shape = None
    return output_shape


def conv1d_signal():
    # Conv1d is used for sequence/signal data
    # Input shape: (batch, channels, length)
    # TODO: Create a Conv1d layer:
    #   1 input channel, 4 output channels, kernel size 3, padding 1
    conv = None
    x = torch.randn(2, 1, 100)  # batch=2, 1 channel, length=100
    # TODO: Pass x through conv and return the output
    output = None
    return output


def conv_weight_shape():
    conv = nn.Conv2d(3, 16, kernel_size=5)
    # TODO: Return the shape of the convolution weights
    # Conv2d weight shape is (out_channels, in_channels, kH, kW)
    shape = None
    return shape


def pooling_layers():
    x = torch.randn(1, 3, 8, 8)
    # TODO: Apply max pooling with kernel_size=2 (halves spatial dims)
    max_pool = None
    max_pooled = None

    # TODO: Apply average pooling with kernel_size=2
    avg_pool = None
    avg_pooled = None

    return max_pooled.shape, avg_pooled.shape


"""
----------------------TESTS-------------------------
------------------DO NOT TOUCH TESTS----------------
"""


def test_create_conv2d():
    conv = create_conv2d()
    assert isinstance(conv, nn.Conv2d)
    assert conv.in_channels == 1
    assert conv.out_channels == 16
    assert conv.kernel_size == (3, 3)
    assert conv.padding == (1, 1)


def test_conv_output_shape():
    shape = conv_output_shape()
    assert shape == torch.Size([1, 16, 28, 28])


def test_conv_with_stride():
    shape = conv_with_stride()
    assert shape == torch.Size([1, 8, 8, 8])


def test_conv1d_signal():
    output = conv1d_signal()
    assert output.shape == torch.Size([2, 4, 100])


def test_conv_weight_shape():
    shape = conv_weight_shape()
    assert shape == torch.Size([16, 3, 5, 5])


def test_pooling_layers():
    max_shape, avg_shape = pooling_layers()
    assert max_shape == torch.Size([1, 3, 4, 4])
    assert avg_shape == torch.Size([1, 3, 4, 4])
