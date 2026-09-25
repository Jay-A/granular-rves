import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

project = "Granular RVEs"
author = "Jay M. Appleton"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
]

source_suffix = {
    ".md": "markdown",
    ".rst": "restructuredtext",
}

master_doc = "index"

html_theme = "pydata_sphinx_theme"
html_title = "Granular RVEs"
