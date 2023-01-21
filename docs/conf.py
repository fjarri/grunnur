# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

import setuptools_scm
from numpy.typing import DTypeLike

sys.path.insert(0, os.path.abspath("../"))


# -- Project information -----------------------------------------------------

project = "grunnur"
copyright = "2020–now, Bogdan Opanchuk"
author = "Bogdan Opanchuk"

# The full version, including alpha/beta/rc tags
release = setuptools_scm.get_version(relative_to=os.path.abspath("../pyproject.toml"))


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
]

autoclass_content = "both"
autodoc_member_order = "groupwise"
autodoc_type_aliases = dict(DTypeLike="DTypeLike")

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Note: set to the lower bound of `numpy` version in the dependencies;
# must be kept synchronized.
intersphinx_mapping = {
    "numpy": ("https://numpy.org/doc/1.22", None),
    "python": ("https://docs.python.org/3", None),
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

nitpick_ignore = [
    ("py:class", "typing.List"),
    ("py:class", "typing.Iterable"),
    ("py:class", "typing.Mapping"),
    ("py:class", "typing.Type"),
    ("py:data", "typing.Optional"),
    ("py:data", "typing.Callable"),
    ("py:data", "typing.Union"),
    ("py:data", "typing.Tuple"),
    ("py:data", "typing.Any"),
    ("py:class", "int"),
    ("py:class", "str"),
    ("py:class", "bool"),
    ("py:class", "numpy.dtype"),
    ("py:class", "numpy.ndarray"),
    ("py:class", "mako.template.Template"),
    ("py:class", "mako.template.DefTemplate"),
]
