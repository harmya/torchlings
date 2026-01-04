def chain_rule_example():
    # TODO: Set requires_grad appropriately
    x = torch.tensor(2.0)
    
    # Forward pass: y = 3x^2 + 2x + 1
    y = ???
    
    # TODO: Compute gradient
    
    # The derivative should be 6x + 2 = 14 when x = 2
    return ???

# Test
def test_chain_rule():
    grad = chain_rule_example()
    assert grad.item() == 14.0