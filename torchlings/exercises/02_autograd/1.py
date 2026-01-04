import torch

def simple_gradient():
    # TODO: Create a tensor with requires_grad=True
    x = torch.tensor([2.0, 3.0, 4.0])
    
    # Compute y = x^2
    y = ???
    
    # Compute the mean of y
    z = ???
    
    # TODO: Compute gradients and return them
    

# Test
def test_gradient():
    grad = simple_gradient()
    expected = torch.tensor([4/3, 2.0, 8/3])
    assert torch.allclose(grad, expected, atol=1e-5)