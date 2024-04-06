# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "climkern"
copyright = "2024, Ty Janoski"
author = "Ty Janoski"
release = "v1.0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon"]

autodoc_default_options = {"members": True}
autoclass_content = "class"
napoleon_numpy_docstring = True
napoleon_google_docstring = False

autodoc_mock_imports = [
    "xarray",
    "cf-xarray",
    "cftime",
    "xesmf",
    "importlib_resources",
    "pooch",
    "tqdm",
    "plac",
    "netCDF4",
]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "alabaster"
html_static_path = ["_static"]
