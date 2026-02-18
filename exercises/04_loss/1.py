# Regression losses: measuring prediction error
import torch
import torch.nn as nn


def mse_loss():
    predictions = torch.tensor([2.5, 0.0, 2.0, 8.0])
    targets = torch.tensor([3.0, -0.5, 2.0, 7.0])
    # TODO: Compute Mean Squared Error loss
    loss = None
    return loss


def l1_loss():
    predictions = torch.tensor([2.5, 0.0, 2.0, 8.0])
    targets = torch.tensor([3.0, -0.5, 2.0, 7.0])
    # TODO: Compute L1 (Mean Absolute Error) loss
    loss = None
    return loss


def huber_loss():
    predictions = torch.tensor([2.5, 0.0, 2.0, 8.0])
    targets = torch.tensor([3.0, -0.5, 2.0, 7.0])
    # TODO: Compute Huber loss (SmoothL1Loss) with default delta=1.0
    # Huber loss is quadratic for small errors, linear for large errors
    loss = None
    return loss


def mse_loss_no_reduction():
    predictions = torch.tensor([1.0, 2.0, 3.0])
    targets = torch.tensor([1.5, 2.5, 3.5])
    # TODO: Compute MSE loss with reduction='none' to get per-element losses
    loss = None
    return loss


def manual_mse():
    predictions = torch.tensor([2.5, 0.0, 2.0, 8.0])
    targets = torch.tensor([3.0, -0.5, 2.0, 7.0])
    # TODO: Compute MSE manually (without nn.MSELoss)
    # MSE = mean((predictions - targets)^2)
    loss = None
    return loss


def weighted_mse():
    predictions = torch.tensor([1.0, 2.0, 3.0, 4.0])
    targets = torch.tensor([1.1, 2.5, 3.0, 4.8])
    weights = torch.tensor([1.0, 2.0, 1.0, 3.0])
    # TODO: Compute weighted MSE: mean(weights * (predictions - targets)^2)
    loss = None
    return loss


"""
----------------------TESTS-------------------------
------------------DO NOT TOUCH TESTS----------------
"""


def test_mse_loss():
    loss = mse_loss()
    expected = torch.tensor(0.375)
    assert torch.allclose(loss, expected, atol=1e-4)


def test_l1_loss():
    loss = l1_loss()
    expected = torch.tensor(0.5)
    assert torch.allclose(loss, expected, atol=1e-4)


def test_huber_loss():
    loss = huber_loss()
    assert loss.ndim == 0  # scalar
    assert loss.item() > 0


def test_mse_loss_no_reduction():
    loss = mse_loss_no_reduction()
    assert loss.shape == torch.Size([3])
    expected = torch.tensor([0.25, 0.25, 0.25])
    assert torch.allclose(loss, expected)


def test_manual_mse():
    manual = manual_mse()
    builtin = nn.MSELoss()(
        torch.tensor([2.5, 0.0, 2.0, 8.0]),
        torch.tensor([3.0, -0.5, 2.0, 7.0]),
    )
    assert torch.allclose(manual, builtin, atol=1e-5)


def test_weighted_mse():
    loss = weighted_mse()
    # weights * (pred - target)^2 = [1*0.01, 2*0.25, 1*0, 3*0.64]
    # = [0.01, 0.5, 0.0, 1.92] -> mean = 0.6075
    expected = torch.tensor(0.6075)
    assert torch.allclose(loss, expected, atol=1e-4)
