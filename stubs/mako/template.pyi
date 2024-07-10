from collections.abc import Sequence
from typing import Any

class Template:
    source: str
    def __init__(
        self,
        *,
        text: str | None = None,
        filename: str | None = None,
        strict_undefined: bool = False,
        imports: Sequence[str] | None = None,
    ): ...
    def get_def(self, name: str) -> DefTemplate: ...
    def list_defs(self) -> list[str]: ...

class DefTemplate:
    def render(self, *args: Any, **kwargs: Any) -> str: ...
