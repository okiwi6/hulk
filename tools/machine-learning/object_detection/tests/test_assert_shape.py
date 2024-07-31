import pytest
import torch
from ssd.utils import assert_ndim, assert_shape


def test_matching_shape():
    tensor = torch.zeros(2, 3, 4, 5)
    assert_shape(tensor, (2, 3, 4, 5))


def test_nonmatching_shape():
    tensor = torch.zeros(2, 3, 4, 5)
    with pytest.raises(AssertionError):
        assert_shape(tensor, (3, 3, 4, 5))


def test_matching_ndim():
    tensor = torch.zeros(2, 1)
    assert_ndim(tensor, 2)


def test_nonmatching_ndim():
    tensor = torch.zeros(3, 2, 1)
    with pytest.raises(AssertionError):
        assert_ndim(tensor, 2)
