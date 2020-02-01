import inspect

from .template import render_template, make_template


def extract_signature_and_value(func_or_str, default_parameters=None):
    if not inspect.isfunction(func_or_str):
        if default_parameters is None:
            parameters = []
        else:
            kind = funcsigs.Parameter.POSITIONAL_OR_KEYWORD
            parameters = [funcsigs.Parameter(name, kind=kind) for name in default_parameters]

        # TODO: since we're using Py>=3.7, use the built-in Signature
        return funcsigs.Signature(parameters), func_or_str

    signature = funcsigs.signature(func_or_str)

    # pass mock values to extract the value
    args = [None] * len(signature.parameters)
    return signature, func_or_str(*args)


def template_from(template):
    """
    Creates a Mako template object from a given string.
    If ``template`` already has ``render()`` method, does nothing.
    """
    if hasattr(template, 'render'):
        return template
    else:
        return make_template(template)


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

    def __init__(self, template_src, render_kwds=None):
        self.template = template_from(template_src)
        self.render_kwds = {} if render_kwds is None else dict(render_kwds)

    @classmethod
    def create(cls, func_or_str, render_kwds=None):
        """
        Creates a snippet from the ``Mako`` def:

        * if ``func_or_str`` is a function, then the def has the same signature as ``func_or_str``,
          and the body equal to the string it returns;
        * if ``func_or_str`` is a string, then the def has empty signature.
        """
        signature, code = extract_signature_and_value(func_or_str)
        return cls(template_def(signature, code), render_kwds=render_kwds)


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

    def __init__(self, template_src, render_kwds=None):
        self.template = template_from(template_src)
        self.render_kwds = {} if render_kwds is None else dict(render_kwds)

    @classmethod
    def create(cls, func_or_str, render_kwds=None):
        """
        Creates a module from the ``Mako`` def:

        * if ``func_or_str`` is a function, then the def has the same signature as ``func_or_str``
          (prefix will be passed as the first positional parameter),
          and the body equal to the string it returns;
        * if ``func_or_str`` is a string, then the def has a single positional argument ``prefix``.
          and the body ``code``.
        """
        signature, code = extract_signature_and_value(func_or_str, default_parameters=['prefix'])
        return cls(template_def(signature, code), render_kwds=render_kwds)


class SourceCollector:

    def __init__(self):
        self.constant_modules = {}
        self.sources = []
        self.prefix_counter = 0

    def add_module(self, module_id, tmpl_def, args, render_kwds):

        # This caching serves two purposes.
        # First, it reduces the amount of generated code by not generating
        # the same module several times.
        # Second, if the same module object is used (without arguments) in other modules,
        # the data structures defined in this module will be suitable
        # for functions in these modules.
        if len(args) == 0:
            if module_id in self.constant_modules:
                return self.constant_modules[module_id]

        prefix = "_mod_" + tmpl_def.name + "_" + str(self.prefix_counter) + "_"
        self.prefix_counter += 1

        src = render_template(tmpl_def, prefix, *args, **render_kwds)
        self.sources.append(src)

        if len(args) == 0:
            self.constant_modules[module_id] = prefix

        return prefix

    def get_source(self):
        return "\n\n".join(self.sources)


class RenderableSnippet:

    def __init__(self, tmpl_def, render_kwds):
        self.template_def = tmpl_def
        self.render_kwds = render_kwds

    def __call__(self, *args):
        return render_template(self.template_def, *args, **self.render_kwds)

    def __str__(self):
        return self()


class RenderableModule:

    def __init__(self, collector, module_id, tmpl_def, render_kwds):
        self.module_id = module_id
        self.collector = collector
        self.template_def = tmpl_def
        self.render_kwds = render_kwds

    def __call__(self, *args):
        return self.collector.add_module(
            self.module_id, self.template_def, args, self.render_kwds)

    def __str__(self):
        return self()


def process(obj, collector):
    if isinstance(obj, Snippet):
        render_kwds = process(obj.render_kwds, collector)
        return RenderableSnippet(obj.template, render_kwds)
    elif isinstance(obj, Module):
        render_kwds = process(obj.render_kwds, collector)
        return RenderableModule(collector, id(obj), obj.template, render_kwds)
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


def render_template_source(src, render_args=None, render_kwds=None):

    if render_args is None:
        render_args = []
    if render_kwds is None:
        render_kwds = {}

    collector = SourceCollector()
    render_args = process(render_args, collector)
    main_renderable = process(Snippet(src, render_kwds=render_kwds), collector)

    main_src = main_renderable(*render_args)

    return collector.get_source() + "\n\n" + main_src
