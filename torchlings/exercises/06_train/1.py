# The training loop: forward, loss, backward, step
import torch
import torch.nn as nn


def single_training_step():
    torch.manual_seed(42)
    model = nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    x = torch.randn(8, 4)
    targets = torch.randint(0, 2, (8,))
    criterion = nn.CrossEntropyLoss()

    # TODO: Perform one training step:
    # 1. Zero the gradients
    # 2. Forward pass
    # 3. Compute loss
    # 4. Backward pass
    # 5. Update parameters
    loss = None

    return loss


def training_reduces_loss():
    torch.manual_seed(0)
    model = nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    x = torch.randn(32, 4)
    targets = torch.randint(0, 2, (32,))

    # TODO: Run 50 training steps on the same data and collect each loss
    losses = []

    return losses


def eval_mode_no_dropout():
    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.Dropout(0.5),
        nn.Linear(8, 2),
    )
    x = torch.randn(4, 4)

    # TODO: Get model output in eval mode (dropout should be disabled)
    # Hint: call model.eval() before forward, then get the output
    output = None
    return output


def train_vs_eval():
    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.BatchNorm1d(8),
        nn.Linear(8, 2),
    )
    x = torch.randn(4, 4)

    # TODO: Get output in train mode
    model.train()
    train_out = None

    # TODO: Get output in eval mode
    model.eval()
    eval_out = None

    return train_out, eval_out


def no_grad_inference():
    torch.manual_seed(42)
    model = nn.Linear(4, 2)
    x = torch.randn(8, 4)
    # TODO: Run inference inside torch.no_grad() context
    # This saves memory by not tracking gradients
    output = None
    return output


"""
----------------------TESTS-------------------------
------------------DO NOT TOUCH TESTS----------------
"""


def test_single_training_step():
    loss = single_training_step()
    assert loss is not None
    assert loss.ndim == 0
    assert loss.item() > 0


def test_training_reduces_loss():
    losses = training_reduces_loss()
    assert len(losses) == 50
    assert losses[-1] < losses[0]


def test_eval_mode_no_dropout():
    output = eval_mode_no_dropout()
    assert output.shape == torch.Size([4, 2])
    # In eval mode, dropout is disabled, so running twice gives same result
    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(4, 8), nn.Dropout(0.5), nn.Linear(8, 2))
    x = torch.randn(4, 4)
    model.eval()
    out2 = model(x)
    assert torch.allclose(output, out2)


def test_train_vs_eval():
    train_out, eval_out = train_vs_eval()
    assert train_out.shape == torch.Size([4, 2])
    assert eval_out.shape == torch.Size([4, 2])


def test_no_grad_inference():
    output = no_grad_inference()
    assert output.shape == torch.Size([8, 2])
    assert not output.requires_grad
