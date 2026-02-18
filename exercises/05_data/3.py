# Collate functions and data transforms
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence


class VariableLengthDataset(Dataset):
    """Dataset where each sample has a different length."""

    def __init__(self):
        self.data = [
            torch.tensor([1.0, 2.0]),
            torch.tensor([3.0, 4.0, 5.0]),
            torch.tensor([6.0]),
            torch.tensor([7.0, 8.0, 9.0, 10.0]),
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def pad_collate_fn(batch):
    # TODO: Implement a collate function that pads variable-length sequences
    # to the same length in a batch
    # Hint: use pad_sequence(batch, batch_first=True, padding_value=0)
    padded = None
    return padded


def collate_with_lengths(batch):
    # TODO: Implement a collate function that returns both the padded batch
    # AND a tensor of original lengths
    # Return a tuple: (padded_batch, lengths_tensor)
    lengths = None
    padded = None
    return padded, lengths


def dict_collate_fn(batch):
    # Each sample is a dict with "features" and "label"
    # TODO: Stack all features into one tensor and all labels into one tensor
    # Return a dict with the stacked tensors
    features = None
    labels = None
    return {"features": features, "labels": labels}


def apply_dataloader_with_collate():
    dataset = VariableLengthDataset()
    # TODO: Create a DataLoader with batch_size=2 and the pad_collate_fn
    loader = None
    # TODO: Get the first batch
    first_batch = None
    return first_batch


class DictDataset(Dataset):
    def __init__(self):
        self.data = [
            {"features": torch.tensor([1.0, 2.0]), "label": torch.tensor(0)},
            {"features": torch.tensor([3.0, 4.0]), "label": torch.tensor(1)},
            {"features": torch.tensor([5.0, 6.0]), "label": torch.tensor(0)},
            {"features": torch.tensor([7.0, 8.0]), "label": torch.tensor(1)},
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def apply_dict_collate():
    dataset = DictDataset()
    # TODO: Create a DataLoader with batch_size=2 and dict_collate_fn
    loader = None
    first_batch = None
    return first_batch


"""
----------------------TESTS-------------------------
------------------DO NOT TOUCH TESTS----------------
"""


def test_pad_collate_fn():
    batch = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0, 5.0])]
    result = pad_collate_fn(batch)
    assert result.shape == torch.Size([2, 3])
    assert result[0, 2].item() == 0.0  # padding


def test_collate_with_lengths():
    batch = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0, 5.0])]
    padded, lengths = collate_with_lengths(batch)
    assert padded.shape == torch.Size([2, 3])
    assert lengths.tolist() == [2, 3]


def test_dict_collate_fn():
    batch = [
        {"features": torch.tensor([1.0, 2.0]), "label": torch.tensor(0)},
        {"features": torch.tensor([3.0, 4.0]), "label": torch.tensor(1)},
    ]
    result = dict_collate_fn(batch)
    assert result["features"].shape == torch.Size([2, 2])
    assert result["labels"].shape == torch.Size([2])


def test_apply_dataloader_with_collate():
    batch = apply_dataloader_with_collate()
    assert batch.shape[0] == 2
    assert batch.shape[1] >= 2  # padded to longest in batch


def test_apply_dict_collate():
    batch = apply_dict_collate()
    assert "features" in batch
    assert "labels" in batch
    assert batch["features"].shape == torch.Size([2, 2])
