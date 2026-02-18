# Custom loss functions: building your own objectives
import torch
import torch.nn as nn
import torch.nn.functional as F


def focal_loss(logits, targets, gamma=2.0):
    # Focal loss down-weights easy examples and focuses on hard ones
    # FL(p_t) = -(1 - p_t)^gamma * log(p_t)
    # TODO: Implement focal loss for binary classification
    # 1. Compute probabilities from logits using sigmoid
    # 2. Get p_t (probability of the true class)
    # 3. Compute focal weight: (1 - p_t)^gamma
    # 4. Compute BCE loss with reduction='none'
    # 5. Apply focal weight and return mean
    probs = None
    p_t = None
    focal_weight = None
    bce = None
    loss = None
    return loss


def margin_ranking_loss():
    # Margin loss: ensures one input is ranked higher than another
    x1 = torch.tensor([1.0, 2.0, 3.0])
    x2 = torch.tensor([2.0, 1.0, 1.0])
    # y=1 means x1 should be ranked higher, y=-1 means x2 should be higher
    y = torch.tensor([1.0, 1.0, -1.0])
    # TODO: Compute margin ranking loss with margin=0.5
    loss = None
    return loss


def cosine_embedding_loss():
    # Measures similarity between pairs using cosine distance
    x1 = torch.randn(3, 5)
    x2 = torch.randn(3, 5)
    # y=1 means similar, y=-1 means dissimilar
    y = torch.tensor([1, -1, 1])
    # TODO: Compute cosine embedding loss with margin=0.5
    loss = None
    return loss


def multi_objective_loss():
    predictions = torch.tensor([1.0, 2.0, 3.0, 4.0])
    targets = torch.tensor([1.1, 2.2, 2.8, 4.1])
    # TODO: Combine MSE and L1 loss with weights: 0.7 * MSE + 0.3 * L1
    mse = None
    l1 = None
    combined = None
    return combined


"""
----------------------TESTS-------------------------
------------------DO NOT TOUCH TESTS----------------
"""


def test_focal_loss():
    logits = torch.tensor([0.5, -0.3, 2.0, -1.0])
    targets = torch.tensor([1.0, 0.0, 1.0, 0.0])
    loss = focal_loss(logits, targets)
    assert loss.ndim == 0
    assert loss.item() > 0
    # Focal loss should be less than standard BCE for easy examples
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    assert loss.item() < bce.item()


def test_margin_ranking_loss():
    loss = margin_ranking_loss()
    assert loss.ndim == 0
    assert loss.item() >= 0


def test_cosine_embedding_loss():
    loss = cosine_embedding_loss()
    assert loss.ndim == 0


def test_multi_objective_loss():
    loss = multi_objective_loss()
    assert loss.ndim == 0
    assert loss.item() > 0
    # Should be between pure MSE and pure L1
    preds = torch.tensor([1.0, 2.0, 3.0, 4.0])
    tgts = torch.tensor([1.1, 2.2, 2.8, 4.1])
    mse = nn.MSELoss()(preds, tgts)
    l1 = nn.L1Loss()(preds, tgts)
    lo = min(mse.item(), l1.item())
    hi = max(mse.item(), l1.item())
    assert lo <= loss.item() <= hi + 1e-5
