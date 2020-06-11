import pytest

from grunnur import Program, CompilationError


def test_compilation_error(mock_context, mock_backend, capsys):
    with mock_backend.make_compilation_fail():
        with pytest.raises(CompilationError):
            Program(mock_context, "")

    captured = capsys.readouterr()
    assert "Failed to compile on device 0" in captured.out

    # check that the full source is shown (including the prelude)
    assert "#define GRUNNUR_" in captured.out
