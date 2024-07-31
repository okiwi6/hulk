import torch


def assert_shape(tensor: torch.Tensor, shape: tuple[int, ...]):
    assert tensor.shape == shape, f"Expected shape {shape} but got {tensor.shape}"


def assert_ndim(tensor: torch.Tensor, ndim: int):
    assert tensor.ndim == ndim, f"Expected {ndim} dimensions, but got {tensor.shape}"
