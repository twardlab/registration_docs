# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Mouse Brain Atlas Registration'
copyright = '2024, Andrew Bennecke'
author = 'Andrew Bennecke and Daniel Tward'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'nbsphinx',
    'sphinxcontrib.htmlhelp'
]

# Configuration for LaTeX output
latex_documents = [
    ('index', 'registration_docs.tex', project, author, 'manual'),
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

import os
import sys
# sys.path.insert(0, os.path.abspath('.'))
sys.path.append('/home/abenneck/Desktop/registration_docs/source')
sys.path.append('/home/runner/work/registration_docs/registration_docs/source')

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'nature'

html_static_path = ['_static']
