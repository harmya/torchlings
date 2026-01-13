# Sequential: composing layers into a pipeline
import torch
import torch.nn as nn


def build_sequential():
    # TODO: Build a sequential model: Linear(10, 5) -> ReLU -> Linear(5, 2)
    model = None
    return model


def sequential_forward():
    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 3),
    )
    x = torch.randn(2, 4)
    # TODO: Pass x through the model
    output = None
    return output


def access_layers():
    model = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 2),
        nn.Sigmoid(),
    )
    # TODO: Access the first Linear layer (index 0)
    first_linear = None
    # TODO: Access the last layer (Sigmoid)
    last_layer = None
    return first_linear, last_layer


def named_sequential():
    # TODO: Build a Sequential using OrderedDict so layers have names:
    # "fc1" -> Linear(10, 5), "act1" -> ReLU, "fc2" -> Linear(5, 1)
    from collections import OrderedDict

    model = None
    return model


def count_model_parameters():
    model = nn.Sequential(
        nn.Linear(20, 10),
        nn.ReLU(),
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 1),
    )
    # TODO: Count total trainable parameters across all layers
    total = None
    return total


"""
----------------------TESTS-------------------------
------------------DO NOT TOUCH TESTS----------------
"""


def test_build_sequential():
    model = build_sequential()
    assert isinstance(model, nn.Sequential)
    assert len(model) == 3
    assert isinstance(model[0], nn.Linear)
    assert isinstance(model[1], nn.ReLU)
    assert isinstance(model[2], nn.Linear)
    assert model[0].in_features == 10
    assert model[2].out_features == 2


def test_sequential_forward():
    output = sequential_forward()
    assert output.shape == torch.Size([2, 3])


def test_access_layers():
    first, last = access_layers()
    assert isinstance(first, nn.Linear)
    assert isinstance(last, nn.Sigmoid)
    assert first.in_features == 10


def test_named_sequential():
    model = named_sequential()
    assert isinstance(model, nn.Sequential)
    assert isinstance(model.fc1, nn.Linear)
    assert isinstance(model.act1, nn.ReLU)
    assert isinstance(model.fc2, nn.Linear)


def test_count_model_parameters():
    total = count_model_parameters()
    # (20*10+10) + (10*5+5) + (5*1+1) = 210 + 55 + 6 = 271
    assert total == 271
