from __future__ import annotations

from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, overload

from ._template import DefTemplate, RenderError
from ._utils import update_dict

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable, Iterable, Mapping, Sequence


SOURCE_COLLECTOR: ContextVar[SourceCollector] = ContextVar("SOURCE_COLLECTOR")


class Snippet:
    """
    Contains a source snippet - a template function that will be rendered in place,
    with possible context that can include other :py:class:`Snippet`
    or :py:class:`Module` objects.
    """

    def __init__(self, template: DefTemplate, render_globals: Mapping[str, Any] = {}):
        """Creates a snippet out of a prepared template."""
        self.name = template.name
        self.template = template
        self.render_globals = render_globals

    def with_added_globals(self, add_globals: Mapping[str, Any] = {}) -> Snippet:
        new_globals = update_dict(
            self.render_globals,
            add_globals,
            error_msg="Cannot add a global '{name}' - it already exists",
        )
        return Snippet(self.template, new_globals)

    @classmethod
    def from_callable(
        cls,
        callable_obj: Callable[..., str],
        name: str = "_snippet",
        render_globals: Mapping[str, Any] = {},
    ) -> Snippet:
        """
        Creates a snippet from a callable returning a string.
        The parameter list of the callable is used to create the pararameter list
        of the resulting template def; the callable should return the body of a
        Mako template def regardless of the arguments it receives.

        :param callable_obj: a callable returning the template source.
        :param name: the snippet's name (will simplify debugging)
        :param render_globals: a dictionary of "globals" to be used when rendering the template.
        """
        template = DefTemplate.from_callable(name, callable_obj)
        return cls(template, render_globals=render_globals)

    @classmethod
    def from_string(
        cls,
        argnames: Iterable[str],
        source: str,
        name: str = "_snippet",
        render_globals: Mapping[str, Any] = {},
    ) -> Snippet:
        """
        Creates a snippet from a template source and a list of arguments.

        :param argnames: names of the arguments for the created template.
        :param source: a string with the template source.
        :param name: the snippet's name (will simplify debugging)
        :param render_globals: a dictionary of "globals" to be used when rendering the template.
        """
        template = DefTemplate.from_string(name, argnames, source)
        return cls(template, render_globals=render_globals)

    def __call__(self, *args: Any) -> str:
        return self.template.render(*args, **self.render_globals)

    def __str__(self) -> str:
        return self()


class Module:
    """
    Contains a source module - a template function that will be rendered at root level,
    and the place where it was called will receive its unique identifier (prefix),
    which is used to prefix all module's functions, types and macros in the global namespace.
    """

    @classmethod
    def from_callable(
        cls,
        callable_obj: Callable[..., str],
        name: str = "_module",
        render_globals: Mapping[str, Any] = {},
    ) -> Module:
        """
        Creates a module from a callable returning a string.
        The parameter list of the callable is used to create the pararameter list
        of the resulting template def; the callable should return the body of a
        Mako template def regardless of the arguments it receives.

        The prefix will be passed as the first argument to the template def on render.

        :param callable_obj: a callable returning the template source.
        :param name: the module's name (will simplify debugging)
        :param render_globals: a dictionary of "globals" to be used when rendering the template.
        """
        template = DefTemplate.from_callable(name, callable_obj)
        return cls(template, render_globals=render_globals)

    @classmethod
    def from_string(
        cls, source: str, name: str = "_module", render_globals: Mapping[str, Any] = {}
    ) -> Module:
        """
        Creates a module from a template source, treated as a body of a
        template def with a single argument (prefix).

        :param source: a string with the template source.
        :param name: the module's name (will simplify debugging)
        :param render_globals: a dictionary of "globals" to be used when rendering the template.
        """
        template = DefTemplate.from_string(name, ["prefix"], source)
        return cls(template, render_globals=render_globals)

    def __init__(self, template: DefTemplate, render_globals: Mapping[str, Any] = {}):
        """
        Creates a module out of a prepared template.

        :param template:
        :param render_globals:
        """
        self.name = template.name
        self.template = template
        self.render_globals = render_globals

    def __call__(self, *args: Any) -> str:
        collector = SOURCE_COLLECTOR.get()
        return collector.add_module(id(self), self.template, args, self.render_globals)

    def __str__(self) -> str:
        return self()


class SourceCollector:
    def __init__(self) -> None:
        self.module_cache: dict[tuple[int, tuple[Any, ...]], str] = {}
        self.sources: list[str] = []
        self.prefix_counter = 0

    def add_module(
        self,
        module_id: int,
        template: DefTemplate,
        args: Sequence[Any],
        render_globals: Mapping[str, Any],
    ) -> str:
        # This caching serves two purposes.
        # First, it reduces the amount of generated code by not generating
        # the same module several times.
        # Second, if the same module object is used in other modules,
        # the data structures defined in this module will be suitable
        # for functions in these modules.
        call_id = (module_id, tuple(args))
        if call_id in self.module_cache:
            return self.module_cache[call_id]

        prefix = "_mod_" + template.name + "_" + str(self.prefix_counter) + "_"
        self.prefix_counter += 1

        src = template.render(prefix, *args, **render_globals)
        self.sources.append(src)

        self.module_cache[call_id] = prefix

        return prefix

    def get_source(self) -> str:
        return "\n".join(self.sources)


def render_with_modules(
    src: str | Callable[..., str] | DefTemplate | Snippet,
    render_args: Sequence[Any] = (),
    render_globals: Mapping[str, Any] = {},
) -> str:
    """
    Renders the given source with given positional arguments and globals.
    If a module is attempted to be rendered, its source is prepended to the resulting source,
    and the caller receives the generated module prefix.

    If ``src`` is a string, a callable or a :py:class:`DefTemplate`,
    a :py:class:`Snippet` is created with a corresponding classmethod or the constructor.

    If ``src`` is a :py:class:`Snippet`, ``render_globals`` will be added to its render globals
    (a ``ValueError`` will be thrown if there is a name clash).

    If any of the nested templates fails to render,
    a :py:class:`~grunnur.template.RenderError` is propagated
    from that place to this function, and raised here.

    :param src: the textual source, template or a snippet to render.
    :param render_args: a list of arguments to pass to the template def.
    :param render_globals: a dict of globals to render the template with.
    """
    if isinstance(src, str):
        if len(render_args) > 0:
            raise ValueError("A textual source cannot have `render_args` set.")
        snippet = Snippet.from_string([], src, name="_main_", render_globals=render_globals)
    elif isinstance(src, DefTemplate):
        snippet = Snippet(src, render_globals=render_globals)
    elif isinstance(src, Snippet):
        snippet = src.with_added_globals(render_globals)
    elif callable(src):
        snippet = Snippet.from_callable(src, name="_main_", render_globals=render_globals)
    else:
        raise TypeError(f"Cannot render an object of type {type(src)}")

    collector = SourceCollector()
    token = SOURCE_COLLECTOR.set(collector)
    try:
        main_src = snippet(*render_args)
    except RenderError as e:
        # The error will come from a chain of modules and snippets rendering each other,
        # so it will be buried deep in the traceback.
        # Setting the cause to None to cut all the intermediate calls which don't carry
        # any important information.
        raise e from None
    finally:
        SOURCE_COLLECTOR.reset(token)

    return collector.get_source() + "\n" + main_src
