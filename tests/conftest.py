"""Shared pytest fixtures for cuDeep tests."""

import pytest


@pytest.fixture
def dtype_float32():
    from cuDeep import DType
    return DType.float32
