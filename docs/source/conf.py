"""
feedbackGRAPE documentation build configuration file, created by
sphinx-quickstart on Mon Jan 01 00:00:00 2025.
"""
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath("../../feedback_grape"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "feedbackGRAPE"
copyright = "2025, Yousef Elbrolosy, Pavlo Bilous, Florian Marquardt"
author = "Yousef Elbrolosy, Pavlo Bilous, Florian Marquardt"
release = "0.0.0b11"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon", "nbsphinx"]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
