import sys
from .template import DefTemplate, RenderError


class Snippet:
    """
    Contains a source snippet.
    See :ref:`tutorial-modules` for details.

    :param template_src: a ``Mako`` template with the module code,
        or a string with the template source.
    :type template_src: ``str`` or ``Mako`` template.
    :param render_kwds: a dictionary which will be used to render the template.
        Can contain other modules and snippets.
    """

    @classmethod
    def from_function(cls, func, name='_snippet', render_kwds={}):
        """
        Creates a snippet from the ``Mako`` def:

        * if ``func_or_str`` is a function, then the def has the same signature as ``func_or_str``,
          and the body equal to the string it returns;
        * if ``func_or_str`` is a string, then the def has empty signature.
        """
        template = DefTemplate.from_function(name, func)
        return cls(template, render_kwds=render_kwds)

    @classmethod
    def from_string(cls, source, name='_snippet', render_kwds={}):
        template = DefTemplate.from_string(name, source)
        return cls(template, render_kwds=render_kwds)

    def __init__(self, template, render_kwds={}):
        self.name = template.name
        self.template = template
        self.render_kwds = render_kwds

    def __process_modules__(self, process):
        return RenderableSnippet(self.template, process(self.render_kwds))


class RenderableSnippet:

    def __init__(self, template, render_kwds):
        self.template = template
        self.render_kwds = render_kwds

    def __call__(self, *args):
        return self.template.render(*args, **self.render_kwds)

    def __str__(self):
        return self()


class Module:
    """
    Contains a source module.
    See :ref:`tutorial-modules` for details.

    :param template_src: a ``Mako`` template with the module code,
        or a string with the template source.
    :type template_src: ``str`` or ``Mako`` template.
    :param render_kwds: a dictionary which will be used to render the template.
        Can contain other modules and snippets.
    """

    @classmethod
    def from_function(cls, func, name='_module', render_kwds={}):
        """
        Creates a module from the ``Mako`` def:

        * if ``func_or_str`` is a function, then the def has the same signature as ``func_or_str``
          (prefix will be passed as the first positional parameter),
          and the body equal to the string it returns;
        * if ``func_or_str`` is a string, then the def has a single positional argument ``prefix``.
          and the body ``code``.
        """
        template = DefTemplate.from_function(name, func)
        return cls(template, render_kwds=render_kwds)

    @classmethod
    def from_string(cls, source, name='_module', render_kwds={}):
        template = DefTemplate.from_string(name, source, argnames=['prefix'])
        return cls(template, render_kwds=render_kwds)

    def __init__(self, template, render_kwds={}):
        self.name = template.name
        self.template = template
        self.render_kwds = render_kwds

    def process(self, collector):
        return RenderableModule(
            collector, id(self), self.template, process(self.render_kwds, collector))


class RenderableModule:

    def __init__(self, collector, module_id, template, render_kwds):
        self.module_id = module_id
        self.collector = collector
        self.template = template
        self.render_kwds = render_kwds

    def __call__(self, *args):
        return self.collector.add_module(self.module_id, self.template, args, self.render_kwds)

    def __str__(self):
        return self()


class SourceCollector:

    def __init__(self):
        self.module_cache = {}
        self.sources = []
        self.prefix_counter = 0

    def add_module(self, module_id, template, args, render_kwds):

        # This caching serves two purposes.
        # First, it reduces the amount of generated code by not generating
        # the same module several times.
        # Second, if the same module object is used in other modules,
        # the data structures defined in this module will be suitable
        # for functions in these modules.
        call_id = (module_id, args)
        if call_id in self.module_cache:
            return self.module_cache[call_id]

        prefix = "_mod_" + template.name + "_" + str(self.prefix_counter) + "_"
        self.prefix_counter += 1

        src = template.render(prefix, *args, **render_kwds)
        self.sources.append(src)

        self.module_cache[call_id] = prefix

        return prefix

    def get_source(self):
        return "\n".join(self.sources)


def process(obj, collector):
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


def render_with_modules(src, render_args=[], render_kwds={}):

    collector = SourceCollector()
    render_args = process(render_args, collector)

    if isinstance(src, str):
        name = "_main_"
        if len(render_args) > 0:
            raise ValueError("A textual source cannot have `render_args` set.")
        snippet = Snippet.from_string(src, name="_main_", render_kwds=render_kwds)
    elif callable(src):
        snippet = Snippet.from_function(src, name="_main_", render_kwds=render_kwds)
    elif isinstance(DefTemplate, src):
        snippet = Snippet(src, render_kwds=render_kwds)
    else:
        raise TypeError(f"Cannot render an object of type {type(src)}")

    main_renderable = process(snippet, collector)

    try:
        main_src = main_renderable(*render_args)
    except RenderError as e:
        print("Failed to render template with")
        print(f"* args: {e.args}\n* kwds: {e.kwds}\n* source:\n{e.source}\n")
        # The error will come from a chain of modules and snippets rendering each other,
        # so it will be buried deep in the traceback.
        # Setting the cause to None to cut all the intermediate calls which don't carry
        # any important information.
        raise e.exception from None

    return collector.get_source() + "\n" + main_src
