# Hooks: inspecting and modifying activations and gradients
import torch
import torch.nn as nn


def forward_hook_capture():
    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 2),
    )
    activations = {}

    # TODO: Register a forward hook on the ReLU layer (model[1])
    # that saves the output in activations["relu"]
    # Hook signature: hook(module, input, output)
    def hook_fn(module, input, output):
        activations["relu"] = output

    # TODO: Register the hook
    handle = None

    x = torch.randn(3, 4)
    model(x)

    # Remove the hook after use
    if handle is not None:
        handle.remove()

    return activations.get("relu")


def backward_hook_capture():
    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 2),
    )
    gradients = {}

    # TODO: Register a full backward hook on the first Linear layer (model[0])
    # that saves the output gradients (grad_output) in gradients["linear"]
    # Hook signature: hook(module, grad_input, grad_output)
    def hook_fn(module, grad_input, grad_output):
        gradients["linear"] = grad_output[0]

    # TODO: Register the hook
    handle = None

    x = torch.randn(3, 4)
    output = model(x)
    output.sum().backward()

    if handle is not None:
        handle.remove()

    return gradients.get("linear")


def forward_hook_modify():
    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 2),
    )

    # TODO: Register a forward hook on the ReLU (model[1]) that
    # doubles the output (multiply by 2)
    # To modify output, return the new value from the hook
    def hook_fn(module, input, output):
        return None  # return modified output

    handle = None

    x = torch.randn(3, 4)
    hooked_output = model(x)

    if handle is not None:
        handle.remove()

    # Run again without hook for comparison
    normal_output = model(x)

    return hooked_output, normal_output


def feature_extraction():
    # Use hooks to extract intermediate features from a model
    model = nn.Sequential(
        nn.Linear(10, 8),
        nn.ReLU(),
        nn.Linear(8, 4),
        nn.ReLU(),
        nn.Linear(4, 2),
    )
    features = {}

    # TODO: Register forward hooks on BOTH ReLU layers to capture
    # their outputs as "relu1" and "relu2"
    handles = []

    x = torch.randn(5, 10)
    model(x)

    for h in handles:
        h.remove()

    return features.get("relu1"), features.get("relu2")


def gradient_scaling_hook():
    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 2),
    )

    # TODO: Register a tensor hook on the output of model[0] (first linear)
    # that scales gradients by 0.1 during backward
    # Use register_hook on the tensor, not the module
    x = torch.randn(3, 4)
    intermediate = model[0](x)
    # TODO: Register hook on intermediate tensor
    # Hint: intermediate.register_hook(lambda grad: ...)

    rest = model[2](torch.relu(intermediate))
    rest.sum().backward()

    return model[0].weight.grad


"""
----------------------TESTS-------------------------
------------------DO NOT TOUCH TESTS----------------
"""


def test_forward_hook_capture():
    activation = forward_hook_capture()
    assert activation is not None
    assert activation.shape == torch.Size([3, 8])
    # ReLU output should have no negative values
    assert torch.all(activation >= 0)


def test_backward_hook_capture():
    grad = backward_hook_capture()
    assert grad is not None
    assert grad.shape == torch.Size([3, 8])


def test_forward_hook_modify():
    hooked, normal = forward_hook_modify()
    # Hooked output should NOT equal normal output (relu was doubled)
    assert not torch.allclose(hooked, normal)


def test_feature_extraction():
    relu1, relu2 = feature_extraction()
    assert relu1 is not None
    assert relu2 is not None
    assert relu1.shape == torch.Size([5, 8])
    assert relu2.shape == torch.Size([5, 4])


def test_gradient_scaling_hook():
    grad = gradient_scaling_hook()
    assert grad is not None
    # Gradient should be scaled down (we can't check exact values due to
    # randomness, but it shouldn't be all zeros)
    assert not torch.allclose(grad, torch.zeros_like(grad))
