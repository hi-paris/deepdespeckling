import sys
import os

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

sys.path.insert(0, os.path.abspath('../'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'deepdespeckling'
copyright = '2024, Hadrien Mariaccia, Emanuele Delsasso'
author = 'Hadrien Mariaccia, Emanuele Delsasso'
release = '0.3'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon'
]

templates_path = ['_templates']
exclude_patterns = ["build", ".venv", ".vscode",
                    "dist", "deepdespeckling.egg-info", "img", "icons"]

language = 'English'

master_doc = "index"

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}

add_module_names = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
