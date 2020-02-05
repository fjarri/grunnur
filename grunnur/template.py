import inspect
import os.path
import re

from mako.template import Template as MakoTemplate


_TEMPLATE_OPTIONS = dict(
    strict_undefined=True,
    imports=['import numpy'])


class RenderError(Exception):

    def __init__(self, exception, args, kwds, source):
        super().__init__()
        self.exception = exception
        self.args = args
        self.kwds = kwds
        self.source = source

    def __str__(self):
        return str(self.exception)


def _render(renderable, source, *args, **kwds):
    try:
        return renderable.render(*args, **kwds)
    except RenderError as e:
        # _render() can be called by a chain of templates which call each other;
        # passing the original render error to the top so that it could be handled there.
        raise
    except Exception as e:
        # TODO: we could collect mako.exceptions.text_error_template().render() here,
        # because ideally it should point to the line where the error occurred,
        # but for some reason it doesn't. So we don't bother for now.
        raise RenderError(e, args, kwds, source)


def _extract_def_source(source, name):
    """
    Attempts to extract the source of a single def from Mako template.
    This makes error messages much more readable.
    """
    match = re.search(
        r"(<%def\s+name\s*=\s*[\"']" + name + r"\(.*>\s*\r?\n.*</%def>)", source, flags=re.DOTALL)
    if match:
        return match.group(1)
    else:
        return source


class Template:

    @classmethod
    def from_associated_file(cls, filename):
        """
        Returns the :py:class:`Template` object created from the file
        which has the same name as ``filename`` and the extension ``.mako``.
        Typically used in computation modules as ``template_for(__filename__)``.
        """
        path, _ext = os.path.splitext(os.path.abspath(filename))
        template_path = path + '.mako'
        mako_template = MakoTemplate(filename=template_path, **_TEMPLATE_OPTIONS)
        return cls(mako_template)

    @classmethod
    def from_string(cls, template_source):
        mako_template = MakoTemplate(text=template_source, **_TEMPLATE_OPTIONS)
        return cls(mako_template)

    def __init__(self, mako_template: MakoTemplate):
        self.name = "<root>"
        self._mako_template = mako_template
        self.source = mako_template.source

    def render(self, *args, **kwds):
        return _render(self._mako_template, self.source, *args, **kwds)

    def get_def(self, name):
        return DefTemplate(name, self._mako_template.get_def(name))


class DefTemplate:

    @classmethod
    def from_function(cls, name, func):
        signature = inspect.signature(func)

        # pass mock values to extract the value
        args = [None] * len(signature.parameters)

        return cls._from_signature_and_body(name, signature, func(*args))

    @classmethod
    def from_string(cls, name, source, argnames=[]):
        kind = inspect.Parameter.POSITIONAL_OR_KEYWORD
        parameters = [inspect.Parameter(name, kind=kind) for name in argnames]
        signature = inspect.Signature(parameters)

        return cls._from_signature_and_body(name, signature, source)

    @classmethod
    def _from_signature_and_body(cls, name, signature, body):
        """
        Returns a ``Mako`` template with the given ``signature``.

        :param signature: a list of postitional argument names,
            or an ``inspect.Signature`` object.
        :code: a body of the template.
        """
        # TODO: pass the source to the DefTemplate constructor directly
        # instead of using _extract_def_source()
        template_src = "<%def name='" + name + str(signature) + "'>\n" + body + "\n</%def>"
        return Template.from_string(template_src).get_def(name)

    def __init__(self, name, mako_def_template):
        self.name = name
        self._mako_def_template = mako_def_template
        self.source = _extract_def_source(mako_def_template.source, name)

    def render(self, *args, **kwds):
        return _render(self._mako_def_template, self.source, *args, **kwds)
