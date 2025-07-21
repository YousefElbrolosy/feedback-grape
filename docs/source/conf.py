"""
feedback_grape documentation build configuration file, created by
sphinx-quickstart on Mon Jan 01 00:00:00 2025.
"""
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "feedback_grape"
copyright = "2025, Yousef Elbrolosy, Pavlo Bilous, Florian Marquardt"
author = "Yousef Elbrolosy, Pavlo Bilous, Florian Marquardt"
# CD: the version of the package
release = "0.0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_design",
    "sphinx.ext.mathjax",
    "myst_nb",
]
myst_enable_extensions = ["dollarmath", "amsmath"]
nb_execution_mode = "off"
nb_remove_cell_tags = ["remove-cell"]
templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_static_path = ['_static']
