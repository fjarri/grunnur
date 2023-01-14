from typing import Any, Optional, Sequence, List

class Template:
    source: str
    def __init__(
        self,
        text: Optional[str] = None,
        filename: Optional[str] = None,
        strict_undefined: bool = False,
        imports: Optional[Sequence[str]] = None,
    ): ...
    def get_def(self, name: str) -> DefTemplate: ...
    def list_defs(self) -> List[str]: ...

class DefTemplate:
    def render(self, *args: Any, **kwargs: Any) -> str: ...
