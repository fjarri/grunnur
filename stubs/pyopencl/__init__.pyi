from typing import List, Any, Sequence

class Platform: ...

class Device:
    platform: Platform

class Context:
    devices: List[Device]

    def __init__(self, devices: Sequence[Device]): ...

class kernel_work_group_info:
    WORK_GROUP_SIZE: Any

def get_platforms() -> List[Platform]: ...
