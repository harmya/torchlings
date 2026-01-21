# Conv architectures: building a small CNN
import torch
import torch.nn as nn
import torch.nn.functional as F


# TODO: Build a simple CNN for image classification
# Architecture:
#   conv1: Conv2d(1, 16, 3, padding=1) -> ReLU -> MaxPool2d(2)
#   conv2: Conv2d(16, 32, 3, padding=1) -> ReLU -> MaxPool2d(2)
#   Flatten, then fc: Linear(32 * 7 * 7, 10)
# This takes 1-channel 28x28 images and classifies into 10 classes
class SimpleCNN(nn.Module):
    pass


def feature_map_shapes():
    # Trace through the spatial dimensions of a CNN
    x = torch.randn(1, 1, 28, 28)
    conv1 = nn.Conv2d(1, 16, 3, padding=1)
    pool = nn.MaxPool2d(2)
    conv2 = nn.Conv2d(16, 32, 3, padding=1)

    # TODO: Pass through conv1 -> pool, record shape
    after_conv1_pool = None

    # TODO: Pass through conv2 -> pool, record shape
    after_conv2_pool = None

    return after_conv1_pool, after_conv2_pool


def global_average_pooling():
    # Global average pooling replaces flattening in modern architectures
    # It averages each feature map down to a single value
    features = torch.randn(4, 64, 8, 8)  # batch=4, 64 channels, 8x8
    # TODO: Apply adaptive average pooling to get (4, 64, 1, 1), then squeeze
    # Result shape should be (4, 64)
    pooled = None
    return pooled


def depthwise_separable_conv():
    # Depthwise separable conv = depthwise conv + pointwise conv
    # Much fewer parameters than standard conv
    x = torch.randn(1, 32, 16, 16)

    # TODO: Create depthwise conv (groups=in_channels, each channel convolved separately)
    # Conv2d(32, 32, 3, padding=1, groups=32)
    depthwise = None

    # TODO: Create pointwise conv (1x1 conv to mix channels)
    # Conv2d(32, 64, 1)
    pointwise = None

    # TODO: Apply depthwise then pointwise
    output = None
    return output


def count_conv_parameters():
    # Compare parameter counts
    # TODO: Count parameters in a standard Conv2d(32, 64, 3, padding=1)
    standard = nn.Conv2d(32, 64, 3, padding=1)
    standard_params = None

    # TODO: Count parameters in depthwise separable equivalent
    dw = nn.Conv2d(32, 32, 3, padding=1, groups=32)
    pw = nn.Conv2d(32, 64, 1)
    separable_params = None

    return standard_params, separable_params


"""
----------------------TESTS-------------------------
------------------DO NOT TOUCH TESTS----------------
"""


def test_simple_cnn():
    model = SimpleCNN()
    x = torch.randn(4, 1, 28, 28)
    output = model(x)
    assert output.shape == torch.Size([4, 10])


def test_feature_map_shapes():
    s1, s2 = feature_map_shapes()
    assert s1 == torch.Size([1, 16, 14, 14])
    assert s2 == torch.Size([1, 32, 7, 7])


def test_global_average_pooling():
    pooled = global_average_pooling()
    assert pooled.shape == torch.Size([4, 64])


def test_depthwise_separable_conv():
    output = depthwise_separable_conv()
    assert output.shape == torch.Size([1, 64, 16, 16])


def test_count_conv_parameters():
    standard, separable = count_conv_parameters()
    # Standard: 32*64*3*3 + 64 = 18496
    # Separable: (32*1*3*3 + 32) + (32*64*1*1 + 64) = 320 + 2112 = 2432
    assert standard == 18496
    assert separable == 2432
