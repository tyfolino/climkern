from setuptools import setup

DISTNAME = 'climkern'
AUTHOR = 'Ty Janoski'
AUTHOR_EMAIL = 'tjanoski@ccny.cuny.edu'
PYTHON_REQUIRES = '>=3.9'

INSTALL_REQUIRES = [
    'xarray>=0.16.2',
    'cf-xarray>=0.5.1',
    'cftime',
    'xesmf>=0.7.1',
    'esmpy',
    'importlib_resources',
    'pooch',
    'tqdm',
    'plac',
    'netCDF4'
]

# get README file
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name=DISTNAME,
    author=AUTHOR,
    version='1.0.0',
    author_email=AUTHOR_EMAIL,
    python_requires=PYTHON_REQUIRES,
    install_requires=INSTALL_REQUIRES,
    long_description=long_description,
    long_description_content_type='text/markdown'
)
