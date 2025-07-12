import torch

def detach_example():
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    
    # Compute y = x^2
    y = x ** 2
    
    # TODO: Create z by detaching y from the computation graph
    z = None
    
    # This should work without error
    w = z ** 2  # z has no gradient history
    
    # TODO: Compute gradient of y with respect to x
    y.sum().backward()
    
    return x.grad, z.requires_grad

# Test
def test_detach():
    grad, requires_grad = detach_example()
    assert grad.tolist() == [2.0, 4.0, 6.0]
    assert requires_grad == False