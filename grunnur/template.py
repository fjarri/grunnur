from __future__ import annotations

import inspect
import os.path
from typing import Callable, Iterable
import re

from mako.template import Template as MakoTemplate
from mako.template import DefTemplate as MakoDefTemplate


_TEMPLATE_OPTIONS = dict(
    strict_undefined=True,
    imports=['import numpy'])


class RenderError(Exception):
    """
    A custom wrapper for Mako template render errors, to facilitate debugging.
    """

    def __init__(self, exception, args, globals_, source):
        super().__init__()
        self.exception = exception
        self.args = args
        self.globals = globals_
        self.source = source

    def __str__(self):
        return str(self.exception)


def _render(renderable, source, *args, **globals_):
    try:
        return renderable.render(*args, **globals_)
    except RenderError as e:
        # _render() can be called by a chain of templates which call each other;
        # passing the original render error to the top so that it could be handled there.
        raise
    except Exception as e:
        # TODO: we could collect mako.exceptions.text_error_template().render() here,
        # because ideally it should point to the line where the error occurred,
        # but for some reason it doesn't. So we don't bother for now.
        raise RenderError(e, args, globals_, source)


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
    """
    A wrapper for mako ``Template`` objects.
    """

    @classmethod
    def from_associated_file(cls, filename: str) -> Template:
        """
        Returns a :py:class:`Template` object created from the file
        which has the same name as ``filename`` and the extension ``.mako``.
        Typically used in computation modules as ``Template.from_associated_file(__filename__)``.
        """
        path, _ext = os.path.splitext(os.path.abspath(filename))
        template_path = path + '.mako'
        mako_template = MakoTemplate(filename=template_path, **_TEMPLATE_OPTIONS)
        return cls(mako_template)

    @classmethod
    def from_string(cls, template_source: str):
        """
        Returns a :py:class:`Template` object created from source.
        """
        mako_template = MakoTemplate(text=template_source, **_TEMPLATE_OPTIONS)
        return cls(mako_template)

    def __init__(self, mako_template: MakoTemplate):
        self.name = "<root>"
        self._mako_template = mako_template
        self.source = mako_template.source

    def render(self, *args, **globals_) -> str:
        return _render(self._mako_template, self.source, *args, **globals_)

    def get_def(self, name: str) -> DefTemplate:
        return DefTemplate(name, self._mako_template.get_def(name))


class DefTemplate:
    """
    A wrapper for Mako ``DefTemplate`` objects.
    """

    @classmethod
    def from_callable(cls, name: str, callable_: Callable[..., str]) -> DefTemplate:
        """
        Creates a template def from a callable returning a string.
        The parameter list of the callable is used to create the pararameter list
        of the resulting template def; the callable should return the body of a
        Mako template def regardless of the arguments it receives.
        """
        signature = inspect.signature(callable_)

        # pass mock values to extract the value
        args = [None] * len(signature.parameters)

        return cls._from_signature_and_body(name, signature, callable_(*args))

    @classmethod
    def from_string(cls, name: str, source: str, argnames: Iterable[str]=[]) -> DefTemplate:
        """
        Creates a template def from a string with its body and a list of argument names.
        """
        kind = inspect.Parameter.POSITIONAL_OR_KEYWORD
        parameters = [inspect.Parameter(name, kind=kind) for name in argnames]
        signature = inspect.Signature(parameters)

        return cls._from_signature_and_body(name, signature, source)

    @classmethod
    def _from_signature_and_body(
            cls, name: str, signature: inspect.Signature, body: str) -> DefTemplate:
        # TODO: pass the source to the DefTemplate constructor directly
        # instead of using _extract_def_source()
        template_src = "<%def name='" + name + str(signature) + "'>\n" + body + "\n</%def>"
        return Template.from_string(template_src).get_def(name)

    def __init__(self, name: str, mako_def_template: MakoDefTemplate):
        self.name = name
        self._mako_def_template = mako_def_template
        self.source = _extract_def_source(mako_def_template.source, name)

    def render(self, *args, **globals_) -> str:
        return _render(self._mako_def_template, self.source, *args, **globals_)
