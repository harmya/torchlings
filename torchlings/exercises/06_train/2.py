# Optimizers: different strategies for updating parameters
import torch
import torch.nn as nn


def sgd_optimizer():
    model = nn.Linear(4, 2)
    # TODO: Create an SGD optimizer with lr=0.01 and momentum=0.9
    optimizer = None
    return optimizer


def adam_optimizer():
    model = nn.Linear(4, 2)
    # TODO: Create an Adam optimizer with lr=0.001
    optimizer = None
    return optimizer


def parameter_groups():
    # Different parts of a model can have different learning rates
    model = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 2),
    )
    # TODO: Create an Adam optimizer with parameter groups:
    #   - First linear layer (model[0]) with lr=0.001
    #   - Second linear layer (model[2]) with lr=0.0001
    optimizer = None
    return optimizer


def weight_decay():
    model = nn.Linear(4, 2)
    # TODO: Create an AdamW optimizer with lr=0.001 and weight_decay=0.01
    # AdamW implements decoupled weight decay (L2 regularization)
    optimizer = None
    return optimizer


def optimizer_state_after_step():
    torch.manual_seed(42)
    model = nn.Linear(4, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    x = torch.randn(8, 4)
    targets = torch.randint(0, 2, (8,))
    criterion = nn.CrossEntropyLoss()

    # Do one step
    optimizer.zero_grad()
    loss = criterion(model(x), targets)
    loss.backward()
    optimizer.step()

    # TODO: Return the optimizer's state_dict
    # This contains both parameter states (momentum buffers, etc.) and hyperparameters
    state = None
    return state


def compare_optimizers():
    # Train the same model with SGD vs Adam and compare convergence
    torch.manual_seed(0)
    x = torch.randn(32, 4)
    targets = torch.randint(0, 2, (32,))
    criterion = nn.CrossEntropyLoss()

    # TODO: Train with SGD (lr=0.1) for 100 steps, collect final loss
    torch.manual_seed(42)
    model_sgd = nn.Linear(4, 2)
    opt_sgd = None
    loss_sgd = None

    # TODO: Train with Adam (lr=0.01) for 100 steps, collect final loss
    torch.manual_seed(42)
    model_adam = nn.Linear(4, 2)
    opt_adam = None
    loss_adam = None

    return loss_sgd, loss_adam


"""
----------------------TESTS-------------------------
------------------DO NOT TOUCH TESTS----------------
"""


def test_sgd_optimizer():
    opt = sgd_optimizer()
    assert isinstance(opt, torch.optim.SGD)
    assert opt.defaults["lr"] == 0.01
    assert opt.defaults["momentum"] == 0.9


def test_adam_optimizer():
    opt = adam_optimizer()
    assert isinstance(opt, torch.optim.Adam)
    assert opt.defaults["lr"] == 0.001


def test_parameter_groups():
    opt = parameter_groups()
    assert len(opt.param_groups) == 2
    assert opt.param_groups[0]["lr"] == 0.001
    assert opt.param_groups[1]["lr"] == 0.0001


def test_weight_decay():
    opt = weight_decay()
    assert isinstance(opt, torch.optim.AdamW)
    assert opt.defaults["weight_decay"] == 0.01


def test_optimizer_state_after_step():
    state = optimizer_state_after_step()
    assert "state" in state
    assert "param_groups" in state
    assert len(state["state"]) > 0


def test_compare_optimizers():
    loss_sgd, loss_adam = compare_optimizers()
    assert loss_sgd.item() > 0
    assert loss_adam.item() > 0
    # Both should have reduced from initial loss
    assert loss_sgd.item() < 1.0
    assert loss_adam.item() < 1.0
