# Graph breaks: what prevents torch.compile from optimizing your code
import torch
import torch.nn as nn


def no_graph_break():
    # This function has NO graph breaks - all operations are torch-traceable
    # TODO: Implement a function that computes: relu(x @ W + b)
    # using only torch operations (no Python control flow on tensor values)
    def fn(x, W, b):
        result = None
        return result

    return fn


def data_dependent_control_flow():
    # Calling .item() or using a tensor value in an if-statement causes a graph break
    # because the compiler can't trace through data-dependent Python control flow

    # BAD: this causes a graph break
    def bad_fn(x):
        if x.sum().item() > 0:  # .item() forces a graph break
            return x * 2
        return x * 3

    # TODO: Rewrite without the graph break using torch.where
    def good_fn(x):
        result = None
        return result

    return bad_fn, good_fn


def avoid_python_list_ops():
    # Building Python lists from tensors inside compiled code causes graph breaks

    # BAD: appending to a Python list
    def bad_fn(x):
        results = []
        for i in range(x.shape[0]):
            results.append(x[i] * 2)
        return torch.stack(results)

    # TODO: Rewrite using vectorized torch operations (no loop needed)
    def good_fn(x):
        result = None
        return result

    return bad_fn, good_fn


def avoid_numpy_conversion():
    # Converting to numpy inside a compiled function causes a graph break

    # BAD: going to numpy and back
    def bad_fn(x):
        np_x = x.numpy()
        return torch.from_numpy(np_x * 2)

    # TODO: Rewrite using only torch operations
    def good_fn(x):
        result = None
        return result

    return bad_fn, good_fn


def static_shapes_matter():
    # torch.compile works best with static (fixed) shapes
    # Dynamic shapes can cause recompilation
    # TODO: Create a function that pads input to a fixed size of 32
    # along dim=0 using torch.nn.functional.pad
    # This avoids dynamic shapes in downstream operations
    def pad_to_fixed(x):
        # x shape: (N, 4) where N <= 32
        # TODO: Pad to (32, 4) with zeros
        import torch.nn.functional as F

        pad_amount = 32 - x.shape[0]
        result = None  # Hint: F.pad(x, (0, 0, 0, pad_amount))
        return result

    return pad_to_fixed


"""
----------------------TESTS-------------------------
------------------DO NOT TOUCH TESTS----------------
"""


def test_no_graph_break():
    fn = no_graph_break()
    W = torch.randn(4, 3)
    b = torch.randn(4)
    x = torch.randn(2, 3)
    result = fn(x, W, b)
    expected = torch.relu(x @ W.T + b)
    assert torch.allclose(result, expected, atol=1e-5)
    # Should compile without graph breaks
    compiled = torch.compile(fn, fullgraph=True)
    compiled_result = compiled(x, W, b)
    assert torch.allclose(compiled_result, expected, atol=1e-5)


def test_data_dependent_control_flow():
    _, good_fn = data_dependent_control_flow()
    x_pos = torch.tensor([1.0, 2.0, 3.0])
    x_neg = torch.tensor([-1.0, -2.0, -3.0])
    assert torch.allclose(good_fn(x_pos), x_pos * 2)
    assert torch.allclose(good_fn(x_neg), x_neg * 3)


def test_avoid_python_list_ops():
    bad_fn, good_fn = avoid_python_list_ops()
    x = torch.tensor([1.0, 2.0, 3.0])
    expected = bad_fn(x)
    result = good_fn(x)
    assert torch.allclose(result, expected)


def test_avoid_numpy_conversion():
    _, good_fn = avoid_numpy_conversion()
    x = torch.tensor([1.0, 2.0, 3.0])
    result = good_fn(x)
    expected = torch.tensor([2.0, 4.0, 6.0])
    assert torch.allclose(result, expected)


def test_static_shapes_matter():
    pad_fn = static_shapes_matter()
    x = torch.randn(10, 4)
    result = pad_fn(x)
    assert result.shape == torch.Size([32, 4])
    assert torch.allclose(result[:10], x)
    assert torch.allclose(result[10:], torch.zeros(22, 4))
