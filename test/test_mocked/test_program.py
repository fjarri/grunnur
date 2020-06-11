import numpy
import pytest

from grunnur import Queue, Array, Program, CompilationError

from ..mock_base import MockKernel, MockSourceSnippet


@pytest.mark.parametrize('no_prelude', [False, True], ids=["with-prelude", "no-prelude"])
def test_compile(mock_context, no_prelude):

    context = mock_context

    src = MockSourceSnippet(kernels=[MockKernel('kernel1')])
    program = Program(context, src, no_prelude=no_prelude)

    if no_prelude:
        assert program.sources[0].prelude.strip() == ""

    a = numpy.arange(16).astype(numpy.int32)
    b = numpy.arange(16).astype(numpy.int32) + 1

    queue = Queue.from_device_idxs(context)

    a_dev = Array.from_host(queue, a)
    b_dev = Array.from_host(queue, b)

    res_dev = Array.empty(queue, 16, numpy.int32)

    program.kernel1(queue, 16, None, res_dev, a_dev, b_dev)


def test_compilation_error(mock_context, mock_backend, capsys):

    src = MockSourceSnippet(should_fail=True)
    with pytest.raises(CompilationError):
        Program(mock_context, src)

    captured = capsys.readouterr()
    assert "Failed to compile on device 0" in captured.out

    # check that the full source is shown (including the prelude)
    assert "#define GRUNNUR_" in captured.out
    assert "<<< mock source >>>" in captured.out
