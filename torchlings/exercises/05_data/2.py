# DataLoader: batching and iterating over data
import torch
from torch.utils.data import DataLoader, TensorDataset


def create_dataloader():
    x = torch.randn(100, 4)
    y = torch.randint(0, 3, (100,))
    dataset = TensorDataset(x, y)
    # TODO: Create a DataLoader with batch_size=16 and shuffle=True
    loader = None
    return loader


def get_first_batch():
    x = torch.arange(20).reshape(20, 1).float()
    y = torch.zeros(20)
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=5, shuffle=False)
    # TODO: Get the first batch from the loader
    # Hint: use next(iter(loader))
    first_batch = None
    return first_batch


def count_batches():
    x = torch.randn(100, 3)
    y = torch.zeros(100)
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    # TODO: Count the number of batches in the loader
    # The last batch may be smaller than batch_size
    num_batches = None
    return num_batches


def drop_last_batch():
    x = torch.randn(100, 3)
    y = torch.zeros(100)
    dataset = TensorDataset(x, y)
    # TODO: Create a DataLoader with batch_size=32 and drop_last=True
    # This drops the last incomplete batch
    loader = None
    # TODO: Count the number of batches
    num_batches = None
    return num_batches


def iterate_epochs():
    x = torch.arange(10).float().unsqueeze(1)
    y = torch.zeros(10)
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=5, shuffle=False)
    # TODO: Iterate through 2 full epochs and collect all batch sizes
    # Each epoch iterates through the entire dataset once
    batch_sizes = []
    return batch_sizes


"""
----------------------TESTS-------------------------
------------------DO NOT TOUCH TESTS----------------
"""


def test_create_dataloader():
    loader = create_dataloader()
    assert isinstance(loader, DataLoader)
    assert loader.batch_size == 16
    batch = next(iter(loader))
    assert batch[0].shape == torch.Size([16, 4])


def test_get_first_batch():
    batch = get_first_batch()
    x_batch, y_batch = batch
    assert x_batch.shape == torch.Size([5, 1])
    assert torch.allclose(x_batch.squeeze(), torch.arange(5).float())


def test_count_batches():
    n = count_batches()
    assert n == 4  # ceil(100/32) = 4


def test_drop_last_batch():
    n = drop_last_batch()
    assert n == 3  # floor(100/32) = 3


def test_iterate_epochs():
    sizes = iterate_epochs()
    assert len(sizes) == 4  # 2 batches per epoch * 2 epochs
    assert all(s == 5 for s in sizes)
