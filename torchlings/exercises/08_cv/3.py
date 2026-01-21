# Data augmentation: making models robust with transforms
import torch
import torch.nn.functional as F


def random_horizontal_flip(image, p=0.5):
    # image shape: (C, H, W)
    # TODO: Flip the image horizontally with probability p
    # Use torch.rand to decide, and torch.flip to flip along the width dim
    torch.manual_seed(0)  # for reproducibility in tests
    if torch.rand(1).item() < p:
        image = None  # flip along width dimension (dim=2 for CHW)
    return image


def random_crop(image, crop_size):
    # image shape: (C, H, W)
    # TODO: Randomly crop a (crop_size, crop_size) patch from the image
    # 1. Compute valid range for top-left corner
    # 2. Randomly select top and left positions
    # 3. Slice the image
    torch.manual_seed(42)
    _, h, w = image.shape
    top = None
    left = None
    cropped = None
    return cropped


def center_crop(image, crop_size):
    # image shape: (C, H, W)
    # TODO: Crop the center (crop_size, crop_size) region
    _, h, w = image.shape
    top = None
    left = None
    cropped = None
    return cropped


def add_gaussian_noise(image, std=0.1):
    # TODO: Add Gaussian noise to the image
    # noise = torch.randn_like(image) * std
    torch.manual_seed(0)
    noisy = None
    return noisy


def resize_image(image, new_h, new_w):
    # image shape: (C, H, W)
    # TODO: Resize using F.interpolate
    # F.interpolate expects (N, C, H, W), so add and remove batch dim
    resized = None
    return resized


def cutout(image, mask_size):
    # Cutout: randomly mask a square region with zeros
    # TODO: Randomly place a (mask_size, mask_size) zero mask on the image
    torch.manual_seed(42)
    _, h, w = image.shape
    result = image.clone()
    top = torch.randint(0, h - mask_size + 1, (1,)).item()
    left = torch.randint(0, w - mask_size + 1, (1,)).item()
    # TODO: Zero out the mask region
    return result


"""
----------------------TESTS-------------------------
------------------DO NOT TOUCH TESTS----------------
"""


def test_random_horizontal_flip():
    image = torch.arange(12).reshape(1, 3, 4).float()
    flipped = random_horizontal_flip(image, p=1.0)
    # Width should be reversed
    assert torch.allclose(flipped[0, 0], torch.tensor([3.0, 2.0, 1.0, 0.0]))


def test_random_crop():
    image = torch.randn(3, 32, 32)
    cropped = random_crop(image, 16)
    assert cropped.shape == torch.Size([3, 16, 16])


def test_center_crop():
    image = torch.randn(3, 32, 32)
    cropped = center_crop(image, 16)
    assert cropped.shape == torch.Size([3, 16, 16])
    # Center crop should get the middle region
    expected = image[:, 8:24, 8:24]
    assert torch.allclose(cropped, expected)


def test_add_gaussian_noise():
    image = torch.ones(3, 8, 8)
    noisy = add_gaussian_noise(image, std=0.1)
    assert noisy.shape == image.shape
    assert not torch.allclose(noisy, image)
    # Noise should be small
    assert (noisy - image).abs().max() < 1.0


def test_resize_image():
    image = torch.randn(3, 32, 32)
    resized = resize_image(image, 64, 64)
    assert resized.shape == torch.Size([3, 64, 64])


def test_cutout():
    image = torch.ones(3, 16, 16)
    result = cutout(image, 4)
    assert result.shape == image.shape
    # Some region should be zero
    assert result.sum() < image.sum()
    # The zero region should be exactly mask_size * mask_size * channels
    num_zeros = (result == 0).sum().item()
    assert num_zeros == 3 * 4 * 4
