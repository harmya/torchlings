# Custom Dataset: implementing the Dataset interface
import torch
from torch.utils.data import Dataset


# TODO: Implement a custom dataset that holds (x, y) pairs
# Must implement __len__ and __getitem__
# __init__ takes a tensor of features (x) and a tensor of labels (y)
# __getitem__ returns a tuple (x[idx], y[idx])
class PairDataset(Dataset):
    pass


# TODO: Implement a dataset that applies a transform to each sample
# __init__ takes a tensor of data and an optional transform function
# __getitem__ returns transform(data[idx]) if transform is set, else data[idx]
class TransformDataset(Dataset):
    pass


def create_pair_dataset():
    x = torch.randn(100, 5)
    y = torch.randint(0, 2, (100,))
    # TODO: Create and return a PairDataset
    dataset = None
    return dataset


def dataset_length():
    x = torch.randn(50, 3)
    y = torch.zeros(50)
    dataset = PairDataset(x, y)
    # TODO: Return the length of the dataset
    length = None
    return length


def dataset_indexing():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y = torch.tensor([0, 1, 0])
    dataset = PairDataset(x, y)
    # TODO: Get the second sample (index 1) from the dataset
    sample = None
    return sample


def dataset_with_transform():
    data = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    # TODO: Create a TransformDataset that squares each element
    transform_fn = lambda x: x**2
    dataset = None
    return dataset


"""
----------------------TESTS-------------------------
------------------DO NOT TOUCH TESTS----------------
"""


def test_create_pair_dataset():
    dataset = create_pair_dataset()
    assert isinstance(dataset, PairDataset)
    assert len(dataset) == 100


def test_dataset_length():
    length = dataset_length()
    assert length == 50


def test_dataset_indexing():
    sample = dataset_indexing()
    assert isinstance(sample, tuple)
    x, y = sample
    assert torch.allclose(x, torch.tensor([3.0, 4.0]))
    assert y.item() == 1


def test_dataset_with_transform():
    dataset = dataset_with_transform()
    assert dataset[0].item() == 1.0
    assert dataset[1].item() == 4.0
    assert dataset[4].item() == 25.0
