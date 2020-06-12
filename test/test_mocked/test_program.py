import numpy
import pytest

from grunnur import Queue, Array, Program, CompilationError, CUDA_API_ID

from ..test_on_device.test_program import (
    _test_compile,
    _test_constant_memory,
    _test_compilation_error,
    )


@pytest.mark.parametrize('no_prelude', [False, True], ids=["with-prelude", "no-prelude"])
def test_compile(mock_context, no_prelude):
    _test_compile(
        context=mock_context,
        no_prelude=no_prelude,
        is_mocked=True)


def test_constant_memory(mock_context):
    _test_constant_memory(
        context=mock_context,
        is_mocked=True)


def test_compilation_error(mock_context, capsys):
    _test_compilation_error(context=mock_context, capsys=capsys, is_mocked=True)
