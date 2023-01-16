from typing import Optional, Sequence

class SourceModule:
    def __init__(
        self,
        source: str,
        options: Optional[Sequence[str]] = None,
        keep: bool = False,
        no_extern_c: bool = False,
        cache_dir: Optional[str] = None,
        include_dirs: Sequence[str] = [],
    ): ...
