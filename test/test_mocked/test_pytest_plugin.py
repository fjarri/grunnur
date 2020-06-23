import pytest

from grunnur import API


def run_tests(options=[]):
    pytest.main([
        "-v",
        "--no-cov",
        "-k", "mocked and pytest_plugin",
        "-m", "selftest"] + options)


@pytest.mark.selftest
def test_api_fixture(api):
    assert isinstance(api, API)


def test_api(mock_backend_factory, capsys):
    backend_pycuda = mock_backend_factory.mock_pycuda()
    backend_pyopencl = mock_backend_factory.mock_pyopencl()

    run_tests()
    captured = capsys.readouterr()
    assert '::test_api_fixture[api(cuda)]' in captured.out
    assert '::test_api_fixture[api(opencl)]' in captured.out

    run_tests(["--api=cuda"])
    captured = capsys.readouterr()
    assert '::test_api_fixture[api(cuda)]' in captured.out
    assert '::test_api_fixture[api(opencl)]' not in captured.out

    run_tests(["--api=opencl"])
    captured = capsys.readouterr()
    assert '::test_api_fixture[api(cuda)]' not in captured.out
    assert '::test_api_fixture[api(opencl)]' in captured.out
