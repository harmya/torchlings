# Normalization: stabilizing training
import torch
import torch.nn as nn


def apply_batch_norm():
    # BatchNorm normalizes across the batch dimension
    # TODO: Create a BatchNorm1d layer for 4 features
    bn = None
    x = torch.randn(8, 4)  # batch of 8, 4 features
    # TODO: Apply batch norm (make sure model is in train mode)
    output = None
    return output


def batch_norm_train_vs_eval():
    bn = nn.BatchNorm1d(4)
    x = torch.randn(8, 4)

    # TODO: Run in training mode and get output
    bn.train()
    train_out = None

    # TODO: Run the SAME input in eval mode and get output
    bn.eval()
    eval_out = None

    # In eval mode, BN uses running statistics instead of batch statistics
    return train_out.shape, eval_out.shape


def apply_layer_norm():
    # LayerNorm normalizes across the feature dimension (no batch dependency)
    # TODO: Create a LayerNorm for features of size [4]
    ln = None
    x = torch.randn(2, 4)
    # TODO: Apply layer norm
    output = None
    return output


def layer_norm_properties():
    ln = nn.LayerNorm(4)
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    output = ln(x)
    # TODO: Compute the mean and std of the output (across the last dim)
    # After layer norm, mean should be ~0 and std should be ~1
    mean = None
    std = None
    return mean, std


def group_norm():
    # GroupNorm divides channels into groups and normalizes within each group
    # TODO: Create a GroupNorm with 4 groups for 8 channels
    gn = None
    x = torch.randn(2, 8, 4, 4)  # batch=2, channels=8, height=4, width=4
    # TODO: Apply group norm
    output = None
    return output


"""
----------------------TESTS-------------------------
------------------DO NOT TOUCH TESTS----------------
"""


def test_apply_batch_norm():
    output = apply_batch_norm()
    assert output.shape == torch.Size([8, 4])


def test_batch_norm_train_vs_eval():
    train_shape, eval_shape = batch_norm_train_vs_eval()
    assert train_shape == torch.Size([8, 4])
    assert eval_shape == torch.Size([8, 4])


def test_apply_layer_norm():
    output = apply_layer_norm()
    assert output.shape == torch.Size([2, 4])


def test_layer_norm_properties():
    mean, std = layer_norm_properties()
    assert torch.allclose(mean, torch.tensor([0.0]), atol=1e-5)
    assert torch.allclose(std, torch.tensor([1.0]), atol=0.2)


def test_group_norm():
    output = group_norm()
    assert output.shape == torch.Size([2, 8, 4, 4])
