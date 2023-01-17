from __future__ import annotations

import inspect
import os.path
from typing import Callable, Iterable, Tuple, Dict, Sequence, Any, Mapping, Optional
import re
import warnings

import mako.template


class RenderError(Exception):
    """
    A custom wrapper for Mako template render errors, to facilitate debugging.
    """

    exception: Exception
    """The original exception thrown by Mako's `render()`."""

    args: Tuple[Any, ...]
    """The arguments used to render the template."""

    globals: Dict[str, Any]
    """The globals used to render the template."""

    source: str
    """The source of the template."""

    def __init__(
        self, exception: Exception, args: Sequence[Any], globals_: Mapping[str, Any], source: str
    ):
        super().__init__()
        self.exception = exception
        self.args = tuple(args)
        self.globals = dict(globals_)
        self.source = source

    def __str__(self) -> str:
        return (
            "Failed to render a template with\n"
            f"* args: {self.args}\n* globals: {self.globals}\n* source:\n{self.source}\n"
            f"* Mako error: ({type(self.exception).__name__}) {self.exception}"
        )


def _extract_def_source(source: str, name: str) -> str:
    """
    Attempts to extract the source of a single def from Mako template.
    This makes error messages much more readable.
    """
    match = re.search(
        r"(<%def\s+name\s*=\s*[\"']" + name + r"\(.*?>.*?</%def>)", source, flags=re.DOTALL
    )
    if not match:
        warnings.warn(f"Could not find the template definition '{name}'", SyntaxWarning)
        return source

    return match.group(1)


def _make_template(
    filename: Optional[str] = None, text: Optional[str] = None
) -> mako.template.Template:
    return mako.template.Template(
        text=text, filename=filename, strict_undefined=True, imports=["import numpy"]
    )


class Template:
    """
    A wrapper for mako ``Template`` objects.
    """

    @classmethod
    def from_associated_file(cls, filename: str) -> "Template":
        """
        Returns a :py:class:`Template` object created from the file
        which has the same name as ``filename`` and the extension ``.mako``.
        Typically used in computation modules as ``Template.from_associated_file(__file__)``.
        """
        path, _ext = os.path.splitext(os.path.abspath(filename))
        template_path = path + ".mako"
        mako_template = _make_template(filename=template_path)
        return cls(mako_template)

    @classmethod
    def from_string(cls, template_source: str) -> "Template":
        """
        Returns a :py:class:`Template` object created from source.
        """
        mako_template = _make_template(text=template_source)
        return cls(mako_template)

    def __init__(self, mako_template: "mako.template.Template"):
        self._mako_template = mako_template
        self._defs: Dict[str, DefTemplate] = {}

    def get_def(self, name: str) -> "DefTemplate":
        """
        Returns the template def with the name ``name``.
        """
        if name in self._defs:
            return self._defs[name]

        if name not in self._mako_template.list_defs():
            raise AttributeError(f"Template has no def '{name}'")

        def_source = _extract_def_source(self._mako_template.source, name)
        def_template = DefTemplate(name, self._mako_template.get_def(name), def_source)
        self._defs[name] = def_template
        return def_template


class DefTemplate:
    """
    A wrapper for Mako ``DefTemplate`` objects.
    """

    @classmethod
    def from_callable(cls, name: str, callable_obj: Callable[..., str]) -> "DefTemplate":
        """
        Creates a template def from a callable returning a string.
        The parameter list of the callable is used to create the pararameter list
        of the resulting template def; the callable should return the body of a
        Mako template def regardless of the arguments it receives.
        """
        signature = inspect.signature(callable_obj)

        # pass mock values to extract the value
        args = [None] * len(signature.parameters)

        return cls._from_signature_and_body(name, signature, callable_obj(*args))

    @classmethod
    def from_string(cls, name: str, argnames: Iterable[str], source: str) -> "DefTemplate":
        """
        Creates a template def from a string with its body and a list of argument names.
        """
        kind = inspect.Parameter.POSITIONAL_OR_KEYWORD
        parameters = [inspect.Parameter(name, kind=kind) for name in argnames]
        signature = inspect.Signature(parameters)

        return cls._from_signature_and_body(name, signature, source)

    @classmethod
    def _from_signature_and_body(
        cls, name: str, signature: inspect.Signature, body: str
    ) -> "DefTemplate":
        src = "<%def name='" + name + str(signature) + "'>\n" + body + "\n</%def>"
        mako_def_template = _make_template(text=src).get_def(name)
        return cls(name, mako_def_template, src)

    def __init__(self, name: str, mako_def_template: "mako.template.DefTemplate", source: str):
        self.name = name
        self._mako_def_template = mako_def_template
        self.source = source

    def render(self, *args: Any, **globals_: Any) -> str:
        """
        Renders the template def with given arguments and globals.
        """
        try:
            return self._mako_def_template.render(*args, **globals_)
        except RenderError as e:
            # _render() can be called by a chain of templates which call each other;
            # passing the original render error to the top so that it could be handled there.
            raise
        except Exception as e:
            # TODO: we could collect mako.exceptions.text_error_template().render() here,
            # because ideally it should point to the line where the error occurred,
            # but for some reason it doesn't. So we don't bother for now.
            raise RenderError(e, args, globals_, self.source)
