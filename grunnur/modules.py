from __future__ import annotations

from typing import Iterable, Mapping, Callable, Union, Tuple, List, Dict

from .template import DefTemplate, RenderError
from .utils import update_dict


class Snippet:
    """
    Contains a source snippet - a template function that will be rendered in place,
    with possible context that can include other :py:class:`Snippet`
    or :py:class:`Module` objects.
    """
    def __init__(self, template: DefTemplate, render_globals: Mapping={}):
        """
        Creates a snippet out of a prepared template.

        :param template:
        :param render_globals:
        """
        self.name = template.name
        self.template = template
        self.render_globals = render_globals

    def with_added_globals(self, add_globals: Mapping={}) -> Snippet:
        new_globals = update_dict(
            self.render_globals, add_globals,
            error_msg="Cannot add a global '{name}' - it already exists")
        return Snippet(self.template, new_globals)

    @classmethod
    def from_callable(
            cls, callable_obj: Callable[..., str],
            name: str='_snippet', render_globals: Mapping={}) -> Snippet:
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
    def from_string(cls, source: str, name: str='_snippet', render_globals: Mapping={}) -> Snippet:
        """
        Creates a snippet from a template source, treated as a body of a
        template def with no arguments.

        :param source: a string with the template source.
        :param name: the snippet's name (will simplify debugging)
        :param render_globals: a dictionary of "globals" to be used when rendering the template.
        """
        template = DefTemplate.from_string(name, [], source)
        return cls(template, render_globals=render_globals)

    def __process_modules__(self, process: Callable) -> RenderableSnippet:
        return RenderableSnippet(self.template, process(self.render_globals))


class RenderableSnippet:
    """
    A snippet with processed dependencies and ready to be rendered.
    """

    def __init__(self, template: DefTemplate, render_globals: Mapping):
        self.template = template
        self.render_globals = render_globals

    def __call__(self, *args) -> str:
        return self.template.render(*args, **self.render_globals)

    def __str__(self):
        return self()


class Module:
    """
    Contains a source module - a template function that will be rendered at root level,
    and the place where it was called will receive its unique identifier (prefix),
    which is used to prefix all module's functions, types and macros in the global namespace.
    """

    @classmethod
    def from_callable(
            cls, callable_obj: Callable[..., str],
            name: str='_module', render_globals: Mapping={}) -> Module:
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
    def from_string(cls, source: str, name: str='_module', render_globals: Mapping={}) -> Module:
        """
        Creates a module from a template source, treated as a body of a
        template def with a single argument (prefix).

        :param source: a string with the template source.
        :param name: the module's name (will simplify debugging)
        :param render_globals: a dictionary of "globals" to be used when rendering the template.
        """
        template = DefTemplate.from_string(name, ['prefix'], source)
        return cls(template, render_globals=render_globals)

    def __init__(self, template: DefTemplate, render_globals: Mapping={}):
        """
        Creates a module out of a prepared template.

        :param template:
        :param render_globals:
        """
        self.name = template.name
        self.template = template
        self.render_globals = render_globals

    def process(self, collector: SourceCollector) -> RenderableModule:
        return RenderableModule(
            collector, id(self), self.template, process(self.render_globals, collector))


class RenderableModule:
    """
    A module with processed dependencies and ready to be rendered.
    """

    def __init__(
            self, collector: SourceCollector, module_id: int,
            template: DefTemplate, render_globals: Mapping):
        self.module_id = module_id
        self.collector = collector
        self.template = template
        self.render_globals = render_globals

    def __call__(self, *args) -> str:
        return self.collector.add_module(self.module_id, self.template, args, self.render_globals)

    def __str__(self):
        return self()


class SourceCollector:

    def __init__(self):
        self.module_cache = {}
        self.sources = []
        self.prefix_counter = 0

    def add_module(
            self, module_id: int, template: DefTemplate,
            args: Iterable, render_globals: Mapping) -> str:

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

    def get_source(self):
        return "\n".join(self.sources)


def process(obj, collector: SourceCollector):
    if isinstance(obj, Module):
        return obj.process(collector)
    elif hasattr(obj, '__process_modules__'):
        return obj.__process_modules__(lambda x: process(x, collector))
    elif isinstance(obj, dict):
        return {k: process(v, collector) for k, v in obj.items()}
    elif isinstance(obj, tuple):
        return tuple(process(v, collector) for v in obj)
    elif isinstance(obj, list):
        return [process(v, collector) for v in obj]
    else:
        return obj


def render_with_modules(
        src: Union[str, Callable[..., str], DefTemplate, Snippet],
        render_args: Union[Tuple, List]=[], render_globals: Dict={}) -> str:
    """
    Renders the given source traversing the arguments and globals processing modules.
    If a module is attempted to be rendered, its source is prepended to the resulting source,
    and the caller receives the generated module prefix.

    Nested arguments/globals of :py:class:`Module` and :py:class:`Snippet`
    are traversed automatically. Also traversed are instances of `dict`, `list` and `tuple`.
    Other classes must define a `__process_modules__(self, process)` method,
    where `process` is a one-argument function traversing any of the supported objects
    and returning the resulting object with all nested :py:class:`Module` objects
    changed to :py:class:`RenderableModule`.

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

    collector = SourceCollector()
    render_args = process(render_args, collector)

    if isinstance(src, str):
        if len(render_args) > 0:
            raise ValueError("A textual source cannot have `render_args` set.")
        snippet = Snippet.from_string(src, name="_main_", render_globals=render_globals)
    elif isinstance(src, DefTemplate):
        snippet = Snippet(src, render_globals=render_globals)
    elif isinstance(src, Snippet):
        snippet = src.with_added_globals(render_globals)
    elif callable(src):
        snippet = Snippet.from_callable(src, name="_main_", render_globals=render_globals)
    else:
        raise TypeError(f"Cannot render an object of type {type(src)}")

    main_renderable = process(snippet, collector)

    try:
        main_src = main_renderable(*render_args)
    except RenderError as e:
        # The error will come from a chain of modules and snippets rendering each other,
        # so it will be buried deep in the traceback.
        # Setting the cause to None to cut all the intermediate calls which don't carry
        # any important information.
        raise e from None

    return collector.get_source() + "\n" + main_src
