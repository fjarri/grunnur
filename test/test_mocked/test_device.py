import pytest

from grunnur import API, Platform, Device, opencl_api_id, cuda_api_id
from grunnur.adapter_base import DeviceType

from ..mock_pycuda import PyCUDADeviceInfo
from ..mock_pyopencl import PyOpenCLDeviceInfo


def test_all(mock_backend):
    mock_backend.add_devices(['Device2', 'Device3'])
    api = API.from_api_id(mock_backend.api_id)
    platform = Platform.from_index(api, 0)
    devices = Device.all(platform)
    device_names = {device.name for device in devices}
    assert device_names == {'Device2', 'Device3'}


@pytest.mark.parametrize('unique_only', [False, True], ids=['all', 'unique'])
def test_all_by_masks(mock_backend, unique_only):
    mock_backend.add_devices(['foo-bar', 'foo-baz', 'bar-baz', 'foo-baz'])
    api = API.from_api_id(mock_backend.api_id)
    platform = Platform.from_index(api, 0)
    devices = Device.all_by_masks(
        platform, include_masks=['foo'], exclude_masks=['bar'], unique_only=unique_only)
    if unique_only:
        assert len(devices) == 1
        assert devices[0].name == 'foo-baz'
    else:
        assert len(devices) == 2
        assert devices[0].name == 'foo-baz'
        assert devices[1].name == 'foo-baz'
        assert devices[0] != devices[1]


def test_from_backend_device(mock_backend):
    mock_backend.add_devices(['Device1'])

    api = API.from_api_id(mock_backend.api_id)

    if api.id == opencl_api_id():
        backend_device = mock_backend.pyopencl.get_platforms()[0].get_devices()[0]
    elif api.id == cuda_api_id():
        backend_device = mock_backend.pycuda_driver.Device(0)
    else:
        raise NotImplementedError

    with pytest.raises(TypeError, match="was not recognized as a device object"):
        Device.from_backend_device(1)

    device = Device.from_backend_device(backend_device)
    assert device.platform.api == api
    if api.id != cuda_api_id():
        assert device.platform.name == 'Platform0'
    assert device.name == 'Device1'


def test_from_index(mock_backend):
    mock_backend.add_devices(['Device1', 'Device2'])
    api = API.from_api_id(mock_backend.api_id)
    platform = Platform.from_index(api, 0)
    device = Device.from_index(platform, 1)
    assert device.name == 'Device2'


def test_params(mock_backend):
    mock_backend.add_devices(['Device1'])
    api = API.from_api_id(mock_backend.api_id)
    platform = Platform.from_index(api, 0)
    device = Device.from_index(platform, 0)

    params1 = device.params
    params2 = device.params
    assert params1 is params2 # check caching


def test_eq(mock_backend):
    mock_backend.add_devices(['Device0', 'Device1'])
    api = API.from_api_id(mock_backend.api_id)

    platform = Platform.from_index(api, 0)
    d0_v1 = Device.from_index(platform, 0)
    d0_v2 = Device.from_index(platform, 0)
    d1 = Device.from_index(platform, 1)

    assert d0_v1 is not d0_v2 and d0_v1 == d0_v2
    assert d0_v1 != d1


def test_hash(mock_backend):
    mock_backend.add_devices(['Device0', 'Device1'])
    api = API.from_api_id(mock_backend.api_id)

    platform = Platform.from_index(api, 0)
    d0 = Device.from_index(platform, 0)
    d1 = Device.from_index(platform, 1)

    d = {d0: 0, d1: 1}
    assert d[d0] == 0
    assert d[d1] == 1


def test_attributes(mock_backend):
    mock_backend.add_devices(['Device1'])
    api = API.from_api_id(mock_backend.api_id)
    p = Platform.from_index(api, 0)
    d = Device.from_index(p, 0)

    assert d.platform == p
    assert d.name == 'Device1'
    assert d.shortcut == p.shortcut + ',0'
    assert str(d) == 'device(' + d.shortcut + ')'


def test_device_parameters_opencl(mock_backend_pyopencl):
    device_info = PyOpenCLDeviceInfo(
        type=DeviceType.GPU,
        max_work_group_size=512,
        max_work_item_sizes=[512, 512, 512],
        local_mem_size=32 * 1024,
        address_bits=32,
        max_compute_units=4)

    mock_backend_pyopencl.add_devices([device_info])
    api = API.from_api_id(mock_backend_pyopencl.api_id)
    d = api.platforms[0].devices[0]

    assert d.params.type == device_info.type
    assert d.params.max_total_local_size == device_info.max_work_group_size
    assert d.params.max_local_sizes == tuple(device_info.max_work_item_sizes)
    assert d.params.warp_size == 32
    mng = 2**device_info.address_bits // device_info.max_work_group_size
    assert d.params.max_num_groups == (mng,) * len(device_info.max_work_item_sizes)
    assert d.params.local_mem_size == device_info.local_mem_size
    assert d.params.local_mem_banks == 32
    assert d.params.compute_units == device_info.max_compute_units


def test_device_parameters_opencl_apple_cpu(mock_backend_pyopencl):
    device_info = PyOpenCLDeviceInfo(
        type=DeviceType.CPU,
        max_work_group_size=512)

    mock_backend_pyopencl.add_platform_with_devices('Apple', [device_info])
    api = API.from_api_id(mock_backend_pyopencl.api_id)
    d = api.platforms[0].devices[0]

    assert d.params.max_total_local_size == 1
    assert d.params.max_local_sizes == (1, 1, 1)


def test_device_parameters_opencl_cpu(mock_backend_pyopencl):
    device_info = PyOpenCLDeviceInfo(
        type=DeviceType.CPU)

    mock_backend_pyopencl.add_devices([device_info])
    api = API.from_api_id(mock_backend_pyopencl.api_id)
    d = api.platforms[0].devices[0]

    assert d.params.local_mem_banks == 1
    assert d.params.warp_size == 1


def test_device_parameters_opencl_cuda1(mock_backend_pyopencl):

    # CUDA 1.x

    device_info = PyOpenCLDeviceInfo(
        extensions=['cl_nv_device_attribute_query'],
        compute_capability_major_nv=1,
        warp_size_nv=7)

    mock_backend_pyopencl.add_devices([device_info])
    api = API.from_api_id(mock_backend_pyopencl.api_id)
    d = api.platforms[0].devices[0]

    assert d.params.local_mem_banks == 16
    assert d.params.warp_size == device_info.warp_size_nv


def test_device_parameters_opencl_cuda2(mock_backend_pyopencl):

    # CUDA 2.x+

    device_info = PyOpenCLDeviceInfo(
        extensions=['cl_nv_device_attribute_query'],
        compute_capability_major_nv=2,
        warp_size_nv=7)

    mock_backend_pyopencl.add_devices([device_info])
    api = API.from_api_id(mock_backend_pyopencl.api_id)
    d = api.platforms[0].devices[0]

    assert d.params.local_mem_banks == 32
    assert d.params.warp_size == device_info.warp_size_nv


def test_device_parameters_opencl_unknown_nv(mock_backend_pyopencl):

    # Unknown nVidia device

    device_info = PyOpenCLDeviceInfo(
        vendor='NVIDIA')

    mock_backend_pyopencl.add_devices([device_info])
    api = API.from_api_id(mock_backend_pyopencl.api_id)
    d = api.platforms[0].devices[0]

    assert d.params.local_mem_banks == 32
    assert d.params.warp_size == 32


def test_device_parameters_cuda(mock_backend_pycuda):
    device_info = PyCUDADeviceInfo(
            max_threads_per_block=512,
            max_block_dim_x=512,
            max_block_dim_y=512,
            max_block_dim_z=32,
            max_grid_dim_x=2**16,
            max_grid_dim_y=2**16,
            max_grid_dim_z=256,
            warp_size=7,
            max_shared_memory_per_block=58*1024,
            multiprocessor_count=3,
            compute_capability=10)

    mock_backend_pycuda.add_devices([device_info])
    api = API.from_api_id(mock_backend_pycuda.api_id)
    d = api.platforms[0].devices[0]

    assert d.params.type == DeviceType.GPU
    assert d.params.max_total_local_size == device_info.max_threads_per_block
    assert d.params.max_local_sizes == (
        device_info.max_block_dim_x,
        device_info.max_block_dim_y,
        device_info.max_block_dim_z)
    assert d.params.warp_size == device_info.warp_size
    assert d.params.max_num_groups == (
        device_info.max_grid_dim_x,
        device_info.max_grid_dim_y,
        device_info.max_grid_dim_z)
    assert d.params.local_mem_size == device_info.max_shared_memory_per_block
    assert d.params.local_mem_banks == 32
    assert d.params.compute_units == device_info.multiprocessor_count


def test_device_parameters_cuda_1(mock_backend_pycuda):

    # CUDA 1.x

    device_info = PyCUDADeviceInfo(
            compute_capability=1)

    mock_backend_pycuda.add_devices([device_info])
    api = API.from_api_id(mock_backend_pycuda.api_id)
    d = api.platforms[0].devices[0]

    assert d.params.local_mem_banks == 16
