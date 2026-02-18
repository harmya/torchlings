# Image tensors: understanding the shape conventions
import torch
import torch.nn as nn


def image_tensor_shape():
    # PyTorch uses NCHW format: (batch, channels, height, width)
    # TODO: Create a random tensor representing a batch of 4 RGB images,
    # each 32x32 pixels
    images = None
    return images


def single_channel_image():
    # TODO: Create a single grayscale image tensor (1 channel, 28x28)
    # with a batch dimension of 1
    image = None
    return image


def extract_channels():
    image = torch.randn(1, 3, 32, 32)
    # TODO: Extract the red, green, and blue channels separately
    # Each should have shape (1, 1, 32, 32)
    red = None
    green = None
    blue = None
    return red, green, blue


def normalize_image():
    # Pixel values typically come in [0, 255] and need to be normalized
    image = torch.randint(0, 256, (1, 3, 32, 32)).float()
    # TODO: Normalize to [0, 1] range
    normalized = None
    return normalized


def channel_mean_std():
    # ImageNet normalization uses per-channel mean and std
    image = torch.rand(1, 3, 224, 224)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    # TODO: Apply standard normalization: (image - mean) / std
    normalized = None
    return normalized


def flatten_for_linear():
    # Conv features need to be flattened before passing to linear layers
    features = torch.randn(8, 64, 4, 4)  # batch=8, 64 channels, 4x4 spatial
    # TODO: Flatten the spatial and channel dimensions, keeping batch dim
    # Result should be shape (8, 1024)
    flat = None
    return flat


"""
----------------------TESTS-------------------------
------------------DO NOT TOUCH TESTS----------------
"""


def test_image_tensor_shape():
    images = image_tensor_shape()
    assert images.shape == torch.Size([4, 3, 32, 32])


def test_single_channel_image():
    image = single_channel_image()
    assert image.shape == torch.Size([1, 1, 28, 28])


def test_extract_channels():
    r, g, b = extract_channels()
    assert r.shape == torch.Size([1, 1, 32, 32])
    assert g.shape == torch.Size([1, 1, 32, 32])
    assert b.shape == torch.Size([1, 1, 32, 32])


def test_normalize_image():
    normed = normalize_image()
    assert normed.min() >= 0.0
    assert normed.max() <= 1.0


def test_channel_mean_std():
    normed = channel_mean_std()
    assert normed.shape == torch.Size([1, 3, 224, 224])
    # After normalization, values should be roughly centered around 0
    assert normed.mean().abs() < 2.0


def test_flatten_for_linear():
    flat = flatten_for_linear()
    assert flat.shape == torch.Size([8, 1024])
