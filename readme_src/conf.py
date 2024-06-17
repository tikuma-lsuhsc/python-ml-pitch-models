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
from typing import get_args

# sys.path.insert(0, os.path.abspath("../src/fcn_f0"))

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import fcn_f0

# -- Project information -----------------------------------------------------

project = "python-fcn-f0"
copyright = (
    "2024, Takeshi (Kesh) Ikuma, Louisiana State University Health Sciences Center",
    "2019 Luc Ardaillon",
    "2018 Jong Wook Kim",
)
author = "Takeshi (Kesh) Ikuma"

release = fcn_f0.__version__

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinxcontrib.restbuilder",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    # "sphinx_autodoc_typehints",
    # "sphinx.ext.intersphinx",
    # "sphinx.ext.autosummary",
    # "sphinx.ext.todo",
    # "sphinxcontrib.blockdiag",
    # "sphinxcontrib.repl",
    # "matplotlib.sphinxext.plot_directive",
]
# Looks for objects in external projects

autodoc_type_aliases = {
    # 'Iterable': 'Iterable',
    "ArrayLike": "ArrayLike",
    # "PretrainedModelName": ", ".join(
    #     (f"'{s}'" for s in get_args(fcn_f0.PretrainedModelName))
    # ),
}

# typehints_fully_qualified = False
# always_document_param_types = False
# always_use_bars_union = False
# typehints_document_rtype = True
# typehints_use_rtype = True
typehints_defaults = "braces"
simplify_optional_unions = True
# typehints_formatter = None
typehints_use_signature = True
typehints_use_signature_return = True


rst_prolog = """
.. role:: python(code)
    :language: python
    :class: highlight
"""

todo_include_todos = True
