from enum import Enum
import weakref


class DeviceType(Enum):
    CPU = 1
    GPU = 2


class MockPyOpenCL:

    def __init__(self, platforms_devices):
        self.pyopencl = Mock_pyopencl(platforms_devices)


class Mock_pyopencl:

    def __init__(self, platforms_devices):

        self._platforms = [
            make_cls(Platform, platform_opts) for platform_opts, _ in platforms_devices]

        # Plaftorms must know about their devices,
        # and devices must know about their parent platforms.
        # To avoid circular references, we retain both device and platform references
        # in the backend object (which will be kept alive by monkeypatch),
        # and use weak references elsewhere.

        self._devices = []
        for pnum, pd in enumerate(platforms_devices):
            platform = self._platforms[pnum]
            _, device_opts = pd
            devices = [make_cls(Device, opts) for opts in device_opts]

            self._devices.append(devices)
            for device in devices:
                device._set_platform(platform)
            platform._set_devices(devices)

        self.device_type = DeviceType

        self.Device = Device
        self.Platform = Platform
        self.Context = Context

    def get_platforms(self):
        return self._platforms


def make_cls(cls, opts):
    if isinstance(opts, (tuple, list)):
        return cls(*opts)
    elif isinstance(opts, dict):
        return cls(**opts)
    else:
        return cls(opts)


class Platform:

    def __init__(self, name):
        self.name = name
        self._devices = None

    def _set_devices(self, devices):
        assert self._devices is None
        self._devices = [weakref.ref(device) for device in devices]

    def get_devices(self):
        return [device() for device in self._devices]


class Device:

    def __init__(self, name, max_work_group_size=1024):
        self.name = name
        self._platform = None

        self.max_work_group_size = max_work_group_size
        self.max_work_item_sizes = [max_work_group_size] * 3
        self.address_bits = 64
        self.type = DeviceType.GPU
        self.extensions = []
        self.vendor = 'Mock Devices'
        self.local_mem_size = 48 * 1024
        self.max_compute_units = 1

    def _set_platform(self, platform):
        assert self._platform is None
        self._platform = weakref.ref(platform)

    @property
    def platform(self):
        return self._platform()


class Context:

    def __init__(self, devices):
        self.devices = devices
