import os
import sys
sys.path.insert(0, os.path.abspath('../src'))

project = 'stochlab'
copyright = '2024, stochlab contributors'
author = 'stochlab contributors'
release = '0.1.1'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'myst_parser',
]

myst_enable_extensions = [
    "dollarmath",
    "amsmath",
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_logo = '_static/logofinalpng-1.png'
html_css_files = ['css/custom.css']
html_theme_options = {
    'logo_only': True,
}

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}

napoleon_numpy_docstring = True
napoleon_google_docstring = False