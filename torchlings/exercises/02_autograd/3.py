import torch

def gradient_accumulation():
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    
    # First forward pass
    y1 = x ** 2
    y1.sum().backward()
    
    # TODO: Store first gradient
    grad1 = None
    
    # Second forward pass without zeroing gradients
    y2 = x ** 3
    y2.sum().backward()
    
    # TODO: Store accumulated gradient
    grad2 = None
    
    # TODO: Zero the gradients
    
    # Third forward pass
    y3 = x
    y3.sum().backward()
    
    # TODO: Store gradient after zeroing
    grad3 = None
    
    return grad1, grad2, grad3

# Test
def test_gradient_accumulation():
    g1, g2, g3 = gradient_accumulation()
    assert g1.tolist() == [2.0, 4.0, 6.0]
    assert g2.tolist() == [3.0, 16.0, 33.0]  # 2x + 3x^2
    assert g3.tolist() == [1.0, 1.0, 1.0]