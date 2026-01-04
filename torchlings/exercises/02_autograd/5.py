# Controlling which parts of computation contribute to gradients

import torch

def selective_gradient_branches():
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    
    # Branch 1: Full gradient flow
    y1 = x ** 2
    
    # TODO: Branch 2: Detach only middle element
    # Create a modified version where x[1] doesn't contribute to gradients
    x_modified = x.clone()
    x_modified[1] = x[1].detach()
    y2 = x_modified ** 3
    
    # TODO: Branch 3: Use torch.no_grad() context for part of computation
    with torch.no_grad():
        temp = x * 2  # This computation won't be tracked
    y3 = temp + x  # Only the addition of x will contribute to gradients
    
    # Combine all branches
    total = y1.sum() + y2.sum() + y3.sum()
    
    # TODO: Compute gradients
    total.backward()
    
    # Expected gradients:
    # x[0]: 2*1 (from y1) + 3*1^2 (from y2) + 1 (from y3) = 2 + 3 + 1 = 6
    # x[1]: 2*2 (from y1) + 0 (detached in y2) + 1 (from y3) = 4 + 0 + 1 = 5
    # x[2]: 2*3 (from y1) + 3*3^2 (from y2) + 1 (from y3) = 6 + 27 + 1 = 34
    
    return x.grad

# Test
def test_selective_gradients():
    grad = selective_gradient_branches()
    expected = torch.tensor([6.0, 5.0, 34.0])
    assert torch.allclose(grad, expected)