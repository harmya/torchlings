# Classification losses: training classifiers
import torch
import torch.nn as nn
import torch.nn.functional as F


def cross_entropy_loss():
    # CrossEntropyLoss expects raw logits (not softmax outputs)
    # logits shape: (batch, num_classes), targets shape: (batch,)
    logits = torch.tensor([[2.0, 1.0, 0.1], [0.5, 2.0, 0.3]])
    targets = torch.tensor([0, 1])  # class indices
    # TODO: Compute cross entropy loss
    loss = None
    return loss


def nll_loss():
    # NLLLoss expects log-probabilities, not raw logits
    logits = torch.tensor([[2.0, 1.0, 0.1], [0.5, 2.0, 0.3]])
    # TODO: First apply log_softmax to logits, then compute NLL loss
    log_probs = None
    targets = torch.tensor([0, 1])
    loss = None
    return loss


def cross_entropy_vs_nll():
    # CrossEntropyLoss = log_softmax + NLLLoss
    logits = torch.tensor([[2.0, 1.0, 0.1], [0.5, 2.0, 0.3]])
    targets = torch.tensor([0, 1])
    # TODO: Compute both and show they are equivalent
    ce_loss = None
    nll_loss = None
    return ce_loss, nll_loss


def binary_cross_entropy():
    # For binary classification, use BCEWithLogitsLoss (numerically stable)
    logits = torch.tensor([0.5, -0.3, 1.2, -0.8])
    targets = torch.tensor([1.0, 0.0, 1.0, 0.0])
    # TODO: Compute binary cross entropy with logits
    loss = None
    return loss


def class_weights():
    # When classes are imbalanced, use weights to give more importance to rare classes
    logits = torch.tensor([[2.0, 1.0, 0.1], [0.5, 2.0, 0.3], [1.0, 0.5, 2.0]])
    targets = torch.tensor([0, 1, 2])
    # TODO: Compute cross entropy with class weights [1.0, 2.0, 3.0]
    # The weight parameter adjusts loss contribution per class
    weights = torch.tensor([1.0, 2.0, 3.0])
    loss = None
    return loss


def label_smoothing():
    logits = torch.tensor([[2.0, 1.0, 0.1], [0.5, 2.0, 0.3]])
    targets = torch.tensor([0, 1])
    # TODO: Compute cross entropy with label_smoothing=0.1
    # Label smoothing mixes the target distribution with a uniform distribution
    loss = None
    return loss


"""
----------------------TESTS-------------------------
------------------DO NOT TOUCH TESTS----------------
"""


def test_cross_entropy_loss():
    loss = cross_entropy_loss()
    assert loss.ndim == 0
    assert loss.item() > 0


def test_nll_loss():
    loss = nll_loss()
    assert loss.ndim == 0
    assert loss.item() > 0


def test_cross_entropy_vs_nll():
    ce, nll = cross_entropy_vs_nll()
    assert torch.allclose(ce, nll, atol=1e-5)


def test_binary_cross_entropy():
    loss = binary_cross_entropy()
    assert loss.ndim == 0
    assert loss.item() > 0


def test_class_weights():
    loss = class_weights()
    assert loss.ndim == 0
    assert loss.item() > 0


def test_label_smoothing():
    loss = label_smoothing()
    assert loss.ndim == 0
    # Label smoothed loss should be slightly higher than standard CE
    standard = nn.CrossEntropyLoss()(
        torch.tensor([[2.0, 1.0, 0.1], [0.5, 2.0, 0.3]]),
        torch.tensor([0, 1]),
    )
    assert loss.item() > standard.item()
