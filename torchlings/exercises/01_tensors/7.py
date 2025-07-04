# Complex tensor transformations: stacking, splitting, and advanced operations
import torch


def stack_tensors():
    tensors = [
        torch.tensor([1, 2, 3]),
        torch.tensor([4, 5, 6]),
        torch.tensor([7, 8, 9]),
    ]
    # TODO: Stack tensors along a new dimension (dim=0 and dim=1)

    stacked_dim0 = None
    stacked_dim1 = None

    return stacked_dim0, stacked_dim1


def split_operations():
    tensor = torch.arange(12).reshape(3, 4)
    # TODO: Split tensor into 3 equal parts along dim=1
    # TODO: Then split into unequal parts: first 1 column, then 2 columns, then 1 column

    equal_splits = None
    unequal_splits = None

    return equal_splits, unequal_splits


def chunk_tensor():
    tensor = torch.arange(20)
    # TODO: Chunk into 4 parts (note: last chunk might be smaller)

    chunks = None
    return chunks


def unbind_operation():
    tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
    # TODO: Unbind along dim=0 and dim=1

    unbind_dim0 = None
    unbind_dim1 = None
    return unbind_dim0, unbind_dim1


def where_operation():
    condition = torch.tensor([[True, False, True], [False, True, False]])
    x = torch.tensor([[1, 2, 3], [4, 5, 6]])
    y = torch.tensor([[10, 20, 30], [40, 50, 60]])
    # TODO: Select from x when condition is True, from y when False and put it in result

    result = None
    return result


def tile_operation():
    tensor = torch.tensor([[1, 2], [3, 4]])
    # TODO: Tile the tensor 2 times along dim=0 and 3 times along dim=1

    result = None
    return result


def movedim_permute():
    tensor = torch.randn(2, 3, 4, 5)
    # TODO: Move dimension 1 to position 3
    # TODO: Then permute to shape (3, 5, 2, 4)

    moved = None
    permuted = None

    return moved, permuted


def diagonal_operations():
    tensor = torch.arange(9).reshape(3, 3)
    # TODO: Extract main diagonal and diagonal with offset=1
    # TODO: Then create a diagonal matrix from a 1D tensor

    main_diag = None
    offset_diag = None
    diag_matrix = None

    return main_diag, offset_diag, diag_matrix


def concatenate_tensors():
    t1 = torch.tensor([[1, 2], [3, 4]])
    t2 = torch.tensor([[5, 6], [7, 8]])
    t3 = torch.tensor([[9, 10], [11, 12]])
    # TODO: Concatenate along dim=0, dim=1,
    # TODO: Then, create a 3D tensor by stacking all three

    cat_dim0 = None
    cat_dim1 = None
    stacked_3d = None

    return cat_dim0, cat_dim1, stacked_3d


"""
----------------------TESTS-------------------------
------------------DO NOT TOUCH TESTS---------------
"""


def test_stack_tensors():
    s0, s1 = stack_tensors()
    assert s0.shape == torch.Size([3, 3])
    assert s1.shape == torch.Size([3, 3])


def test_split_operations():
    eq, uneq = split_operations()
    assert len(eq) == 4
    assert all(
        s.shape == torch.Size([3, 1]) or s.shape == torch.Size([3, 2]) for s in eq
    )
    assert len(uneq) == 3
    assert uneq[0].shape == torch.Size([3, 1])
    assert uneq[1].shape == torch.Size([3, 2])


def test_chunk_tensor():
    chunks = chunk_tensor()
    assert len(chunks) == 4
    assert chunks[0].shape == torch.Size([5])


def test_unbind_operation():
    u0, u1 = unbind_operation()
    assert len(u0) == 2
    assert len(u1) == 3


def test_where_operation():
    where_result = where_operation()
    assert where_result.tolist() == [[1, 20, 3], [40, 5, 60]]


def test_tile_operation():
    tiled = tile_operation()
    assert tiled.shape == torch.Size([4, 6])


def test_movedim_permute():
    moved, perm = movedim_permute()
    assert moved.shape == torch.Size([2, 4, 5, 3])
    assert perm.shape == torch.Size([3, 5, 2, 4])


def test_diagonal_operations():
    main_d, off_d, diag_m = diagonal_operations()
    assert main_d.tolist() == [0, 4, 8]
    assert off_d.tolist() == [1, 5]
    assert diag_m.shape == torch.Size([4, 4])


def test_concatenate_tensors():
    c0, c1, s3d = concatenate_tensors()
    assert c0.shape == torch.Size([6, 2])
    assert c1.shape == torch.Size([2, 6])
    assert s3d.shape == torch.Size([3, 2, 2])
