import pytest

from grunnur import API, Platform, Device, Context


def run_tests(options=[]):
    pytest.main([
        '-v',
        '--no-cov',
        '-m', 'plugin_inner_test'] + options)


@pytest.mark.plugin_inner_test
def test_api_fixture(api):
    assert isinstance(api, API)


def test_api(mock_backend_factory, capsys):
    backend_pycuda = mock_backend_factory.mock_pycuda()
    backend_pycuda.add_devices(['Device1'])
    backend_pyopencl = mock_backend_factory.mock_pyopencl()
    backend_pyopencl.add_platform_with_devices('Bar', ['Device2'])

    run_tests(['-k', 'test_api_fixture'])
    captured = capsys.readouterr()
    assert 'device(cuda,0,0): nVidia CUDA, Device1' in captured.out
    assert 'device(opencl,0,0): Bar, Device2' in captured.out
    assert '::test_api_fixture[api(cuda)]' in captured.out
    assert '::test_api_fixture[api(opencl)]' in captured.out

    run_tests(['--api=cuda', '-k', 'test_api_fixture'])
    captured = capsys.readouterr()
    assert 'device(cuda,0,0): nVidia CUDA, Device1' in captured.out
    assert 'device(opencl,0,0): Bar, Device2' not in captured.out
    assert '::test_api_fixture[api(cuda)]' in captured.out
    assert '::test_api_fixture[api(opencl)]' not in captured.out

    run_tests(['--api=opencl', '-k', 'test_api_fixture'])
    captured = capsys.readouterr()
    assert 'device(cuda,0,0): nVidia CUDA, Device1' not in captured.out
    assert 'device(opencl,0,0): Bar, Device2' in captured.out
    assert '::test_api_fixture[api(cuda)]' not in captured.out
    assert '::test_api_fixture[api(opencl)]' in captured.out


def test_no_api(mock_backend_factory, capsys):
    run_tests(['-k', 'test_api_fixture'])
    captured = capsys.readouterr()
    assert '::test_api_fixture[no_api]' in captured.out


@pytest.mark.plugin_inner_test
def test_platform_fixture(platform):
    assert isinstance(platform, Platform)


def test_platform(mock_backend_factory, capsys):
    backend_pyopencl = mock_backend_factory.mock_pyopencl()
    backend_pyopencl.add_platform_with_devices('Foo', ['Device1'])
    backend_pyopencl.add_platform_with_devices('Bar', ['Device2'])

    run_tests(['-k', 'test_platform_fixture'])
    captured = capsys.readouterr()
    assert 'device(opencl,0,0): Foo, Device1' in captured.out
    assert 'device(opencl,1,0): Bar, Device2' in captured.out
    assert '::test_platform_fixture[platform(opencl,0)]' in captured.out
    assert '::test_platform_fixture[platform(opencl,1)]' in captured.out

    run_tests(['--platform-include-mask=Bar', '-k', 'test_platform_fixture'])
    captured = capsys.readouterr()
    assert 'device(opencl,0,0): Foo, Device1' not in captured.out
    assert 'device(opencl,1,0): Bar, Device2' in captured.out
    assert '::test_platform_fixture[platform(opencl,0)]' not in captured.out
    assert '::test_platform_fixture[platform(opencl,1)]' in captured.out

    run_tests(['--platform-exclude-mask=Bar', '-k', 'test_platform_fixture'])
    captured = capsys.readouterr()
    assert 'device(opencl,0,0): Foo, Device1' in captured.out
    assert 'device(opencl,1,0): Bar, Device2' not in captured.out
    assert '::test_platform_fixture[platform(opencl,0)]' in captured.out
    assert '::test_platform_fixture[platform(opencl,1)]' not in captured.out


def test_no_platform(mock_backend_factory, capsys):
    run_tests(['-k', 'test_platform_fixture'])
    captured = capsys.readouterr()
    assert '::test_platform_fixture[no_platform]' in captured.out


@pytest.mark.plugin_inner_test
def test_device_fixture(device):
    assert isinstance(device, Device)


def test_device(mock_backend_factory, capsys):
    backend_pyopencl = mock_backend_factory.mock_pyopencl()
    backend_pyopencl.add_platform_with_devices('Foo', ['Device1', 'Device2'])
    backend_pyopencl.add_platform_with_devices('Bar', ['Device2', 'Device3'])

    run_tests(['-k', 'test_device_fixture'])
    captured = capsys.readouterr()
    assert 'device(opencl,0,0): Foo, Device1' in captured.out
    assert 'device(opencl,0,1): Foo, Device2' in captured.out
    assert 'device(opencl,1,0): Bar, Device2' in captured.out
    assert 'device(opencl,1,1): Bar, Device3' in captured.out
    assert '::test_device_fixture[device(opencl,0,0)]' in captured.out
    assert '::test_device_fixture[device(opencl,0,1)]' in captured.out
    assert '::test_device_fixture[device(opencl,1,0)]' in captured.out
    assert '::test_device_fixture[device(opencl,1,1)]' in captured.out

    run_tests(['--device-include-mask=Device2', '-k', 'test_device_fixture'])
    captured = capsys.readouterr()
    assert 'device(opencl,0,0): Foo, Device1' not in captured.out
    assert 'device(opencl,0,1): Foo, Device2' in captured.out
    assert 'device(opencl,1,0): Bar, Device2' in captured.out
    assert 'device(opencl,1,1): Bar, Device3' not in captured.out
    assert '::test_device_fixture[device(opencl,0,0)]' not in captured.out
    assert '::test_device_fixture[device(opencl,0,1)]' in captured.out
    assert '::test_device_fixture[device(opencl,1,0)]' in captured.out
    assert '::test_device_fixture[device(opencl,1,1)]' not in captured.out

    run_tests(['--device-exclude-mask=Device2', '-k', 'test_device_fixture'])
    captured = capsys.readouterr()
    assert 'device(opencl,0,0): Foo, Device1' in captured.out
    assert 'device(opencl,0,1): Foo, Device2' not in captured.out
    assert 'device(opencl,1,0): Bar, Device2' not in captured.out
    assert 'device(opencl,1,1): Bar, Device3' in captured.out
    assert '::test_device_fixture[device(opencl,0,0)]' in captured.out
    assert '::test_device_fixture[device(opencl,0,1)]' not in captured.out
    assert '::test_device_fixture[device(opencl,1,0)]' not in captured.out
    assert '::test_device_fixture[device(opencl,1,1)]' in captured.out


def test_duplicate_devices(mock_backend_factory, capsys):
    backend_pyopencl = mock_backend_factory.mock_pyopencl()
    backend_pyopencl.add_platform_with_devices('Foo', ['Device1', 'Device1', 'Device2'])

    run_tests(['-k', 'test_device_fixture'])
    captured = capsys.readouterr()
    assert 'device(opencl,0,0): Foo, Device1' in captured.out
    assert 'device(opencl,0,1): Foo, Device1' not in captured.out
    assert 'device(opencl,0,2): Foo, Device2' in captured.out
    assert '::test_device_fixture[device(opencl,0,0)]' in captured.out
    assert '::test_device_fixture[device(opencl,0,1)]' not in captured.out
    assert '::test_device_fixture[device(opencl,0,2)]' in captured.out

    run_tests(['--include-duplicate-devices', '-k', 'test_device_fixture'])
    captured = capsys.readouterr()
    assert 'device(opencl,0,0): Foo, Device1' in captured.out
    assert 'device(opencl,0,1): Foo, Device1' in captured.out
    assert 'device(opencl,0,2): Foo, Device2' in captured.out
    assert '::test_device_fixture[device(opencl,0,0)]' in captured.out
    assert '::test_device_fixture[device(opencl,0,1)]' in captured.out
    assert '::test_device_fixture[device(opencl,0,2)]' in captured.out


def test_no_device(mock_backend_factory, capsys):
    run_tests(['-k', 'test_device_fixture'])
    captured = capsys.readouterr()
    assert 'No GPGPU devices available' in captured.out
    assert '::test_device_fixture[no_device]' in captured.out


@pytest.mark.plugin_inner_test
def test_context_fixture(context):
    assert isinstance(context, Context)
    assert len(context.devices) == 1


def test_context(mock_backend_factory, capsys):
    backend_pyopencl = mock_backend_factory.mock_pyopencl()
    backend_pyopencl.add_platform_with_devices('Foo', ['Device1', 'Device2'])

    run_tests(['-k', 'test_context_fixture'])
    captured = capsys.readouterr()
    assert 'device(opencl,0,0): Foo, Device1' in captured.out
    assert 'device(opencl,0,1): Foo, Device2' in captured.out
    assert '::test_context_fixture[opencl,0,0]' in captured.out
    assert '::test_context_fixture[opencl,0,1]' in captured.out

    run_tests(['--device-include-mask=Device1', '-k', 'test_context'])
    captured = capsys.readouterr()
    assert 'device(opencl,0,0): Foo, Device1' in captured.out
    assert 'device(opencl,0,1): Foo, Device2' not in captured.out
    assert '::test_context_fixture[opencl,0,0]' in captured.out
    assert '::test_context_fixture[opencl,0,1]' not in captured.out


def test_no_context(mock_backend_factory, capsys):
    run_tests(['-k', 'test_context_fixture'])
    captured = capsys.readouterr()
    assert '::test_context_fixture[no_device]' in captured.out


@pytest.mark.plugin_inner_test
def test_multi_device_context_fixture(multi_device_context):
    assert isinstance(multi_device_context, Context)
    assert len(multi_device_context.devices) > 1


# FIXME: decide on the exact logic in this case.
def test_multi_device_context(mock_backend_factory, capsys):
    backend_pyopencl = mock_backend_factory.mock_pyopencl()
    # Two of the devices have the same names to check that they will be picked up
    backend_pyopencl.add_platform_with_devices('Foo', ['Device1', 'Device1', 'Device3'])

    run_tests(['-k', 'test_multi_device_context_fixture'])
    captured = capsys.readouterr()
    assert 'device(opencl,0,0): Foo, Device1' in captured.out
    # Multi-device context does not currently include all the devices used to the list
    assert 'device(opencl,0,1): Foo, Device1' not in captured.out
    assert 'device(opencl,0,2): Foo, Device3' in captured.out
    assert '::test_multi_device_context_fixture[opencl,0,0+opencl,0,1+opencl,0,2]' in captured.out


def test_no_multi_device_context(mock_backend_factory, capsys):
    backend_pyopencl = mock_backend_factory.mock_pyopencl()
    backend_pyopencl.add_platform_with_devices('Foo', ['Device1'])

    run_tests(['-k', 'test_multi_device_context_fixture'])
    captured = capsys.readouterr()
    assert '::test_multi_device_context_fixture[no_multi_device]' in captured.out
