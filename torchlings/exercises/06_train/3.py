# Learning rate schedulers: adjusting the learning rate during training
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, LambdaLR


def step_lr_schedule():
    model = nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    # TODO: Create a StepLR scheduler that decays lr by 0.1 every 10 steps
    scheduler = None

    # TODO: Collect the learning rate after each of 30 steps
    # (call optimizer.step() and scheduler.step() each iteration)
    lrs = []

    return lrs


def cosine_annealing():
    model = nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    # TODO: Create a CosineAnnealingLR scheduler with T_max=50
    scheduler = None

    # TODO: Collect learning rates for 50 steps
    lrs = []

    return lrs


def warmup_schedule():
    model = nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    warmup_steps = 10
    total_steps = 50

    # TODO: Create a LambdaLR scheduler that:
    #   - Linearly warms up from 0 to base_lr over warmup_steps
    #   - Then stays at base_lr for remaining steps
    # The lambda receives the step number and returns a multiplier
    scheduler = None

    # TODO: Collect learning rates for total_steps
    lrs = []

    return lrs


def get_last_lr():
    model = nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

    for _ in range(10):
        optimizer.step()
        scheduler.step()

    # TODO: Return the current learning rate from the scheduler
    current_lr = None
    return current_lr


"""
----------------------TESTS-------------------------
------------------DO NOT TOUCH TESTS----------------
"""


def test_step_lr_schedule():
    lrs = step_lr_schedule()
    assert len(lrs) == 30
    assert abs(lrs[0] - 0.1) < 1e-6
    assert abs(lrs[10] - 0.01) < 1e-6
    assert abs(lrs[20] - 0.001) < 1e-6


def test_cosine_annealing():
    lrs = cosine_annealing()
    assert len(lrs) == 50
    assert abs(lrs[0] - 0.1) < 1e-4
    # Cosine should decrease then reach near 0 at T_max
    assert lrs[-1] < lrs[0]


def test_warmup_schedule():
    lrs = warmup_schedule()
    assert len(lrs) == 50
    # Should start near 0 and increase
    assert lrs[0] < lrs[9]
    # After warmup, should be at base lr
    assert abs(lrs[10] - 0.1) < 1e-5


def test_get_last_lr():
    lr = get_last_lr()
    # After 10 steps with step_size=5, gamma=0.5: 0.1 * 0.5^2 = 0.025
    assert abs(lr - 0.025) < 1e-6
