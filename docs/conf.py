import os
import sys
sys.path.insert(0, os.path.abspath('../src'))

project = 'stochlab'
copyright = '2024, stochlab contributors'
author = 'stochlab contributors'
release = '0.1.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'myst_parser',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}

napoleon_numpy_docstring = True
napoleon_google_docstring = False