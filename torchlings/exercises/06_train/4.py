# Checkpointing: saving and loading model state
import torch
import torch.nn as nn
import tempfile
import os


def save_model_state(model, path):
    # TODO: Save only the model's state_dict to the given path
    pass


def load_model_state(path):
    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 2),
    )
    # TODO: Load the state dict from path into the model
    # Return the model with loaded weights
    return model


def save_full_checkpoint(model, optimizer, epoch, loss, path):
    # TODO: Save a full training checkpoint containing:
    #   - model_state_dict
    #   - optimizer_state_dict
    #   - epoch
    #   - loss
    pass


def load_full_checkpoint(path):
    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 2),
    )
    optimizer = torch.optim.Adam(model.parameters())
    # TODO: Load the checkpoint and restore model, optimizer, epoch, and loss
    checkpoint = None
    epoch = None
    loss = None
    return model, optimizer, epoch, loss


def state_dict_keys():
    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 2),
    )
    # TODO: Return the list of keys in the model's state_dict
    keys = None
    return keys


"""
----------------------TESTS-------------------------
------------------DO NOT TOUCH TESTS----------------
"""


def test_save_and_load_model():
    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    x = torch.randn(2, 4)
    original_output = model(x)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    try:
        save_model_state(model, path)
        loaded = load_model_state(path)
        loaded_output = loaded(x)
        assert torch.allclose(original_output, loaded_output)
    finally:
        os.unlink(path)


def test_full_checkpoint():
    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Do a training step so optimizer has state
    x = torch.randn(4, 4)
    loss = model(x).sum()
    loss.backward()
    optimizer.step()

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    try:
        save_full_checkpoint(model, optimizer, epoch=5, loss=0.123, path=path)
        loaded_model, loaded_opt, epoch, loss_val = load_full_checkpoint(path)
        assert epoch == 5
        assert abs(loss_val - 0.123) < 1e-5
    finally:
        os.unlink(path)


def test_state_dict_keys():
    keys = state_dict_keys()
    assert "0.weight" in keys
    assert "0.bias" in keys
    assert "2.weight" in keys
    assert "2.bias" in keys
    assert len(keys) == 4
