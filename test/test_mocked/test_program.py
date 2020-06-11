import pytest

from grunnur import Program, CompilationError

from ..mock_base import MockKernel, MockSourceSnippet


def test_compile(mock_context):
    k1 = MockKernel('kernel1')
    k2 = MockKernel('kernel2')
    src = MockSourceSnippet(kernels=[k1, k2])
    p = Program(mock_context, src)

    assert p.kernel1
    assert p.kernel2


def test_compilation_error(mock_context, mock_backend, capsys):

    src = MockSourceSnippet(should_fail=True)
    with pytest.raises(CompilationError):
        Program(mock_context, src)

    captured = capsys.readouterr()
    assert "Failed to compile on device 0" in captured.out

    # check that the full source is shown (including the prelude)
    assert "#define GRUNNUR_" in captured.out
